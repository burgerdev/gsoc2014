#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
import vigra
import logging

from collections import defaultdict
from functools import partial, wraps
from itertools import count as InfiniteLabelIterator

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpCompressedCache
from lazyflow.request import Request, RequestPool
#from lazyflow.request import RequestLock as ReqLock
# the lazyflow lock seems to have deadlock issues sometimes
from threading import Lock as ReqLock
from threading import Condition as _Condition

from lazycc import UnionFindArray

# logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

_LABEL_TYPE = np.uint32


class Condition(object):
    def __init__(self):
        self._cond = _Condition()
        self._count = 0

    # register a thread working on this chunk
    def register(self):
        with self._cond:
            self._count += 1

    # wait until all threads working on this chunk are finished
    def wait(self):
        with self._cond:
            while self._count > 0:
                self._cond.wait()

    # signal that the current thread is finished working on this chunk
    def signal(self):
        with self._cond:
            self._count = max(self._count-1, 0)
            self._cond.notify_all()


# locking decorator that locks per chunk
def _chunksynchronized(method):
    @wraps(method)
    def synchronizedmethod(self, chunkIndex, *args, **kwargs):
        with self._chunk_locks[chunkIndex]:
            return method(self, chunkIndex, *args, **kwargs)
    return synchronizedmethod


# decorator that tracks how many threads are labeling a chunk
def _supervised(method):
    @wraps(method)
    def supervisedmethod(self, chunkIndex, *args, **kwargs):
        self._currentlyLabeling[chunkIndex].register()
        a = method(self, chunkIndex, *args, **kwargs)
        self._currentlyLabeling[chunkIndex].signal()
        return a
    return supervisedmethod


# general approach
# ================
#
# There are 3 kinds of labels that we need to consider throughout the operator:
#     * local labels: The output of the chunk wise labelVolume calls. These are
#       stored in self._cache, a compressed VigraArray.
#       aka 'local'
#     * global indices: The mapping of local labels to unique global indices.
#       The actual implemetation is hidden in self.localToGlobal().
#       aka 'global'
#     * global labels: The final labels that are communicated to the outside
#       world. These must be contiguous, i.e. if  global label j appears in the
#       output, then for every global label i<j i also appears in the output.
#       The actual implementation is hidden in self.globalToFinal().
#       aka 'final'
#
# 3d data only, xyz
class OpLazyCC(Operator):
    Input = InputSlot()
    ChunkShape = InputSlot()
    Output = OutputSlot()

    # debug outputs
    _NonGlobalOutput = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLazyCC, self).__init__(*args, **kwargs)
        self._lock = ReqLock()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = _LABEL_TYPE
        self._NonGlobalOutput.meta.assignFrom(self.Output.meta)
        assert self.Input.meta.dtype in [np.uint8, np.uint32, np.uint64],\
            "Cannot label data type {}".format(self.Input.meta.dtype)

        self._setDefaultInternals()

    def execute(self, slot, subindex, roi, result):
        if slot is self.Output:
            chunks = self._roiToChunkIndex(roi)
            for chunk in chunks:
                self._finalize(chunk)

            self._mapArray(roi, result)
        elif slot is self._NonGlobalOutput:
            chunks = self._roiToChunkIndex(roi)
            for chunk in chunks:
                self._finalize(chunk)

            self._mapArray(roi, result, global_labels=False)

    def propagateDirty(self, slot, subindex, roi):
        # TODO: this is already correct, but may be over-zealous
        # we would have to label each chunk that was set dirty and check
        # for changed labels. Therefore, we would have to check if the
        # dirty region is 'small enough', etc etc.
        self._setDefaultInternals()
        self.Output.setDirty(slice(None))

    # grow the requested region such that all labels inside that region are
    # final
    # @param chunkIndex the index of the chunk to finalize
    # @param labels array of labels that need to be finalized (omit for all)
    @_supervised
    def _finalize(self, chunkIndex, labels=None):
        #logger.info("Finalizing {} ...".format(chunkIndex))

        # get a list of neighbours that have to be checked
        neighbours = self._generateNeighbours(chunkIndex)

        # label this chunk first (not as a request, because most of the
        # time this chunk will already be labeled)
        self._label(chunkIndex)

        def processNeighbour(chunk):
            self._label(chunk)
            self._merge(*self._orderPair(chunkIndex, chunk))

        # label all neighbouring chunks, merge adjacent labels
        for other in neighbours:
            processNeighbour(other)

        if labels is None:
            # this means that we need to finalize all labels that show up in
            # the chunk
            labels = self.localToGlobal(chunkIndex, mapping=False,
                                        update=False)

        # tell others that we are finalizing this chunk
        labels, otherLabels = self._registerFinalizedGlobalIndices(chunkIndex,
                                                                   labels,
                                                                   neighbours)

        for i, l in zip(neighbours, otherLabels):
            # get the label overlap between this chunk and its neighbour
            d = np.intersect1d(labels, l)
            # don't go to finalized neighbours
            # no need for lock
            if self._finalizedIndices[i] is not None:
                finalized = map(self._uf.findLabel, self._finalizedIndices[i])
                d = np.setdiff1d(d, finalized)
            d = d.astype(_LABEL_TYPE)
            if len(d) > 0:
                #logger.debug("Going from {} to {} because of labels {}".format(chunkIndex, i, d))
                # start DFS recursion
                self._finalize(i, labels=d)

    def _registerFinalizedGlobalIndices(self, chunkIndex, labels, neighbours):
        if self._finalizedIndices[chunkIndex] is None:
            self._finalizedIndices[chunkIndex] = np.zeros((0,),
                                                          dtype=_LABEL_TYPE)

        # let the others know that we are finalizing this chunk
        # and compute the updated labels on the way
        with self._lock:
            # currently finalized global indices for this chunk, possibly not
            # up to date
            currently_finalized = self._finalizedIndices[chunkIndex]

            # update both sets
            finalized = map(self._uf.findIndex, currently_finalized)
            labels = map(self._uf.findIndex, labels)

            # now that we have the lock, lets globalize the neighbouring labels
            otherLabels = [self.localToGlobal(n, mapping=False)
                           for n in neighbours]

            # update the finalization state
            now_finalized = np.union1d(finalized, labels).astype(_LABEL_TYPE)
            self._finalizedIndices[chunkIndex] = now_finalized

        labels = np.setdiff1d(labels, finalized).astype(_LABEL_TYPE)
        return labels, otherLabels

    # label a chunk and store information
    @_chunksynchronized
    def _label(self, chunkIndex):
        if self._numIndices[chunkIndex] >= 0:
            # this chunk is already labeled
            return

        # get the raw data
        roi = self._chunkIndexToRoi(chunkIndex)
        inputChunk = self.Input.get(roi).wait()

        # label the raw data
        #logger.info("Labeling {} ...".format(chunkIndex))
        labeled = vigra.analysis.labelVolumeWithBackground(inputChunk)
        del inputChunk

        # store the labeled data in cache
        self._cache[roi.toSlice()] = labeled

        # update the labeling information
        numLabels = labeled.max()  # we ignore 0 here
        self._numIndices[chunkIndex] = numLabels
        if numLabels > 0:
            with self._lock:
                # get 1 label that determines the offset
                offset = self._uf.makeNewIndex()
                # the offset is such that label 1 in the local chunk maps to
                # 'offset' in the global context
                self._globalLabelOffset[chunkIndex] = offset - 1

                # get n-1 more labels
                for i in range(numLabels-1):
                    self._uf.makeNewIndex()

    # merge the labels of two adjacent chunks
    @_chunksynchronized
    def _merge(self, chunkA, chunkB):
        if chunkB in self._mergeMap[chunkA]:
            return
        #logger.info("Merging {} {} ...".format(chunkA, chunkB))
        self._mergeMap[chunkA].append(chunkB)

        hyperplane_roi_a, hyperplane_roi_b = \
            self._chunkIndexToHyperplane(chunkA, chunkB)
        hyperplane_index_a = hyperplane_roi_a.toSlice()
        hyperplane_index_b = hyperplane_roi_b.toSlice()

        label_hyperplane_a = self._cache[hyperplane_index_a]
        label_hyperplane_b = self._cache[hyperplane_index_b]

        # see if we have border labels at all
        adjacent_bool_inds = np.logical_and(label_hyperplane_a > 0,
                                            label_hyperplane_b > 0)
        if not np.any(adjacent_bool_inds):
            return

        hyperplane_a = self.Input[hyperplane_index_a].wait()
        hyperplane_b = self.Input[hyperplane_index_b].wait()
        adjacent_bool_inds = np.logical_and(adjacent_bool_inds,
                                            hyperplane_a == hyperplane_b)

        # union find manipulations are critical
        map_a = self.localToGlobal(chunkA, mapping=True)
        map_b = self.localToGlobal(chunkB, mapping=True)
        labels_a = map_a[label_hyperplane_a[adjacent_bool_inds]]
        labels_b = map_b[label_hyperplane_b[adjacent_bool_inds]]
        with self._lock:
            for a, b in zip(labels_a, labels_b):
                self._uf.makeUnion(a, b)

    # get a rectangular region with final global labels
    # @param roi region of interest
    # @param result array of shape roi.stop - roi.start, will be filled
    def _mapArray(self, roi, result, global_labels=True):
        # TODO perhaps with pixeloperator?
        assert np.all(roi.stop - roi.start == result.shape)
        indices = self._roiToChunkIndex(roi)
        for idx in indices:
            # first of all, wait until all threads are finished with this chunk
            # if we get past the next line, all other threads also got past it
            # and we can delete the condition object to save some space (the
            # defaultdict is thread safe, at least in CPython)
            self._currentlyLabeling[idx].wait()
            #del self._currentlyLabeling[idx]

            newroi = self._chunkIndexToRoi(idx)
            newroi.stop = np.minimum(newroi.stop, roi.stop)
            newroi.start = np.maximum(newroi.start, roi.start)
            labels = self.localToGlobal(idx, mapping=True)
            if global_labels:
                self.globalToFinal(labels)
            chunk = self._cache[newroi.toSlice()]
            newroi.start -= roi.start
            newroi.stop -= roi.start
            s = newroi.toSlice()
            result[s] = labels[chunk]

    # returns an array of global labels in use by this chunk if 'mapping' is
    # False, a mapping of local labels to global labels otherwise
    # if update is set to False, the labels will correspond to the originally
    # assigned global labels, otherwise you will get the most recent results
    # of UnionFind
    def localToGlobal(self, chunkIndex, mapping=True, update=True):
        offset = self._globalLabelOffset[chunkIndex]
        numLabels = self._numIndices[chunkIndex]
        labels = np.arange(1, numLabels+1, dtype=_LABEL_TYPE) + offset

        if update:
            labels = np.asarray(map(self._uf.findIndex, labels),
                                dtype=_LABEL_TYPE)

        if not mapping:
            return labels
        else:
            # we got 'numLabels' real labels, and one label '0', so our
            # output has to have numLabels+1 elements
            out = np.zeros((numLabels+1,), dtype=_LABEL_TYPE)
            out[1:] = labels
            return out

    # map an array of global indices to final labels
    def globalToFinal(self, labels):
        for l in np.unique(labels):
            if l == 0:
                continue

            # adding a global label is critical
            with self._lock:
                if l not in self._globalLabels:
                    nextLabel = self._labelIterator.next()
                    #logger.info("Adding {}->{} to global Labels {}".format(l, nextLabel, self._globalLabels))
                    #logger.info("  Note: {} == {} == {}".format(l, self._uf.findIndex(l), self._uf.findLabel(l)))
                    self._globalLabels[l] = nextLabel
            labels[labels == l] = self._globalLabels[l]

    ##########################################################################
    ##################### HELPER METHODS #####################################
    ##########################################################################

    # create roi object from chunk index
    def _chunkIndexToRoi(self, index):
        shape = self.Input.meta.shape
        start = self._chunkShape * np.asarray(index)
        stop = self._chunkShape * (np.asarray(index) + 1)
        stop = np.where(stop > shape, shape, stop)
        roi = SubRegion(self.Input,
                        start=tuple(start), stop=tuple(stop))
        return roi

    # create a list of chunk indices needed for a particular roi
    def _roiToChunkIndex(self, roi):
        cs = self._chunkShape
        start = np.asarray(roi.start)
        stop = np.asarray(roi.stop)
        start_cs = start / cs
        stop_cs = stop / cs
        # add one if division was not even
        stop_cs += np.where(stop % cs, 1, 0)
        chunks = []
        for x in range(start_cs[0], stop_cs[0]):
            for y in range(start_cs[1], stop_cs[1]):
                for z in range(start_cs[2], stop_cs[2]):
                    chunks.append((x, y, z))
        return chunks

    # compute the adjacent hyperplanes of two chunks (1 pix wide)
    # @return 2-tuple of roi's for the respective chunk
    def _chunkIndexToHyperplane(self, chunkA, chunkB):
        rev = False
        for i in range(len(chunkA)):
            if chunkA[i] > chunkB[i]:
                rev = True
                chunkA, chunkB = chunkB, chunkA
            if chunkA[i] < chunkB[i]:
                roiA = self._chunkIndexToRoi(chunkA)
                roiB = self._chunkIndexToRoi(chunkB)
                start = np.asarray(roiA.start)
                start[i] = roiA.stop[i] - 1
                roiA.start = tuple(start)
                stop = np.asarray(roiB.stop)
                stop[i] = roiB.start[i] + 1
                roiB.stop = tuple(stop)
        if rev:
            return roiB, roiA
        else:
            return roiA, roiB

    # generate a list of adjacent chunks
    def _generateNeighbours(self, chunkIndex):
        n = []
        idx = np.asarray(chunkIndex, dtype=np.int)
        for i in range(len(chunkIndex)):
            if idx[i] > 0:
                new = idx.copy()
                new[i] -= 1
                n.append(tuple(new))
            if idx[i]+1 < self._chunkArrayShape[i]:
                new = idx.copy()
                new[i] += 1
                n.append(tuple(new))
        return n

    # fills attributes with standard values, call on each setupOutputs
    def _setDefaultInternals(self):

        # chunk array shape calculation
        shape = self.Input.meta.shape
        chunkShape = self.ChunkShape.value
        assert len(shape) == len(chunkShape),\
            "Encountered an invalid chunkShape"
        f = lambda i: shape[i]//chunkShape[i] + (1 if shape[i] % chunkShape[i]
                                                 else 0)
        self._chunkArrayShape = tuple(map(f, range(3)))
        self._chunkShape = np.asarray(chunkShape, dtype=np.int)

        ### local labels ###
        # cache for local labels
        self._cache = vigra.ChunkedArrayCompressed(shape, dtype=_LABEL_TYPE)

        ### global indices ###
        # offset (global labels - local labels) per chunk
        self._globalLabelOffset = np.ones(self._chunkArrayShape,
                                          dtype=_LABEL_TYPE)
        # keep track of number of indices in chunk (-1 == not labeled yet)
        self._numIndices = -np.ones(self._chunkArrayShape, dtype=np.int32)

        # union find data structure, tells us for every global index to which
        # label it belongs
        self._uf = UnionFindArray(_LABEL_TYPE(1))

        ### global labels ###
        # keep track of assigned global labels
        self._labelIterator = InfiniteLabelIterator(1)
        self._globalLabels = dict()

        ### algorithmic ###

        # keep track of merged regions
        self._mergeMap = defaultdict(list)

        # keep track of the indices that have been finalized
        self._finalizedIndices = np.empty(self._chunkArrayShape,
                                          dtype=np.object)

        # count how many threads are labeling this chunk
        self._currentlyLabeling = defaultdict(Condition)

        # locks that keep threads from changing a specific chunk
        locks = [ReqLock() for i in xrange(self._numIndices.size)]
        locks = np.asarray(locks, dtype=np.object)
        self._chunk_locks = locks.reshape(self._numIndices.shape)

    # order a pair of chunk indices lexicographically
    # (ret[0] is top-left-in-front-of of ret[1])
    @staticmethod
    def _orderPair(tupA, tupB):
        for a, b in zip(tupA, tupB):
            if a < b:
                return tupA, tupB
            if a > b:
                return tupB, tupA
        logger.warn("tupA and tupB are the same")
        return tupA, tupB
