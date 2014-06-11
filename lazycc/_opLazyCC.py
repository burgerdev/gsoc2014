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
from lazyflow.request import RequestLock as RLock

from lazycc import UnionFindArray

# logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

_LABEL_TYPE = np.uint32


# locking decorator that locks per chunk
def _chunksynchronized(method):
    @wraps(method)
    def synchronizedmethod(self, chunkIndex, *args, **kwargs):
        with self._chunk_locks[chunkIndex]:
            return method(self, chunkIndex, *args, **kwargs)
    return synchronizedmethod


# 3d data only, xyz
class OpLazyCC(Operator):
    Input = InputSlot()
    ChunkShape = InputSlot()
    Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLazyCC, self).__init__(*args, **kwargs)
        self._lock = RLock()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = _LABEL_TYPE
        assert self.Input.meta.dtype in [np.uint8, np.uint32, np.uint64],\
            "Cannot label data type {}".format(self.Input.meta.dtype)

        # keep track of assigned global labels
        self._labelIterator = InfiniteLabelIterator(1)
        self._globalLabels = dict()

        # chunk array shape calculation
        shape = self.Input.meta.shape
        chunkShape = self.ChunkShape.value
        assert len(shape) == len(chunkShape),\
            "Encountered an invalid chunkShape"
        f = lambda i: shape[i]//chunkShape[i] + (1 if shape[i] % chunkShape[i]
                                                 else 0)
        self._chunkArrayShape = tuple(map(f, range(3)))
        self._chunkShape = np.asarray(chunkShape, dtype=np.int)

        # keep track of number of labels in chunk (-1 == not labeled yet)
        self._numLabels = -np.ones(self._chunkArrayShape, dtype=np.int32)
        locks = [RLock() for i in xrange(self._numLabels.size)]
        locks = np.asarray(locks, dtype=np.object)
        self._chunk_locks = locks.reshape(self._numLabels.shape)

        # keep track of the labels that have been finalized
        self._finalizedLabels = np.empty(self._chunkArrayShape,
                                         dtype=np.object)

        # offset (global labels - local labels) per chunk
        self._globalLabelOffset = np.ones(self._chunkArrayShape,
                                          dtype=_LABEL_TYPE)

        # global union find data structure
        self._uf = UnionFindArray(_LABEL_TYPE(1))

        # keep track of merged regions
        self._mergeMap = defaultdict(list)

        # cache for local labels
        self._cache = vigra.ChunkedArrayCompressed(shape, dtype=_LABEL_TYPE)

    def execute(self, slot, subindex, roi, result):
        assert slot is self.Output, "Invalid request to execute"

        # this has not proven to be any better
        '''
        pool = RequestPool()

        chunks = self._roiToChunkIndex(roi)
        for i in np.random.permutation(np.arange(len(chunks), dtype=np.int)):
            pool.add(Request(partial(self._finalize, chunks[i])))

        pool.wait()
        pool.clean()
        '''
        chunks = self._roiToChunkIndex(roi)
        for chunk in chunks:
            self._finalize(chunk)

        self._mapArray(roi, result)

    def propagateDirty(self, slot, subindex, roi):
        # TODO: this is already correct, but may be over-zealous
        # we would have to label each chunk that was set dirty and check
        # for changed labels. Therefore, we would have to check if the
        # dirty region is 'small enough', etc etc.
        # HACK
        self.setupOutputs()
        self.Output.setDirty(slice(None))

    # grow the requested region such that all labels inside that region are
    # final
    # @param chunkIndex the index of the chunk to finalize
    # @param labels array of labels that need to be finalized (omit for all)
    def _finalize(self, chunkIndex, labels=None):
        logger.info("Finalizing {} ...".format(chunkIndex))

        # get a list of neighbours that have to be checked
        neighbours = self._generateNeighbours(chunkIndex)

        # label this chunk first (not as a request, because most of the
        # time this chunk will already be labeled)
        self._label(chunkIndex)

        def processNeighbour(chunk):
            self._label(chunk)
            self._merge(*self._orderPair(chunkIndex, chunk))

        # this is apparently also not that good
        '''
        pool = RequestPool()
        for otherChunk in neighbours:
            pool.add(Request(partial(processNeighbour, otherChunk)))
        pool.wait()
        pool.clean()
        '''
        for other in neighbours:
            processNeighbour(other)

        if labels is None:
            labels = self._getLabelsForChunk(chunkIndex)

        if self._finalizedLabels[chunkIndex] is None:
            self._finalizedLabels[chunkIndex] = np.zeros((0,),
                                                         dtype=_LABEL_TYPE)

        # let the others know that we are finalizing this chunk
        # and compute the updated labels on the way
        with self._lock:
            finalized = map(self._uf.findLabel, self._finalizedLabels[chunkIndex])
            labels = map(self._uf.findLabel, labels)
            # now that we have the lock, lets globalize the neighbouring labels
            otherLabels = \
                map(lambda x: map(self._uf.findLabel, self._getLabelsForChunk(x)),
                    neighbours)

        now_finalized = np.union1d(finalized, labels).astype(_LABEL_TYPE)
        self._finalizedLabels[chunkIndex] = now_finalized

        labels = np.setdiff1d(labels, finalized)

        for i, l in zip(neighbours, otherLabels):
            d = np.intersect1d(labels, l)
            # don't go to finalized neighbours
            if self._finalizedLabels[i] is not None:
                finalized = map(self._uf.findLabel, self._finalizedLabels[i])
                d = np.setdiff1d(d, finalized)
            d = d.astype(_LABEL_TYPE)
            if len(d) > 0:
                logger.debug("Going from {} to {} because of labels {}".format(chunkIndex, i, d))
                # start DFS recursion
                self._finalize(i, labels=d)

    # label a chunk and store information
    @_chunksynchronized
    def _label(self, chunkIndex):
        if self._numLabels[chunkIndex] >= 0:
            # this chunk is already labeled
            return

        # get the raw data
        roi = self._chunkIndexToRoi(chunkIndex)
        inputChunk = self.Input.get(roi).wait()

        # label the raw data
        logger.info("Labeling {} ...".format(chunkIndex))
        labeled = vigra.analysis.labelVolumeWithBackground(inputChunk)
        del inputChunk

        # store the labeled data in cache
        self._cache[roi.toSlice()] = labeled

        # update the labeling information
        numLabels = labeled.max()  # we ignore 0 here
        self._numLabels[chunkIndex] = numLabels
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
        logger.info("Merging {} {} ...".format(chunkA, chunkB))
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
        map_a = self._getLabelsForChunk(chunkA, mapping=True)
        map_b = self._getLabelsForChunk(chunkB, mapping=True)
        labels_a = map_a[label_hyperplane_a[adjacent_bool_inds]]
        labels_b = map_b[label_hyperplane_b[adjacent_bool_inds]]
        with self._lock:
            for a, b in zip(labels_a, labels_b):
                self._uf.makeUnion(a, b)

    # get a rectangular region with final global labels
    # @param roi region of interest
    # @param result array of shape roi.stop - roi.start, will be filled
    def _mapArray(self, roi, result):
        # TODO perhaps with pixeloperator?
        assert np.all(roi.stop - roi.start == result.shape)
        indices = self._roiToChunkIndex(roi)
        for idx in indices:
            newroi = self._chunkIndexToRoi(idx)
            newroi.stop = np.minimum(newroi.stop, roi.stop)
            newroi.start = np.maximum(newroi.start, roi.start)
            labels = self._getLabelsForChunk(idx, mapping=True)
            self._toGlobal(labels)
            chunk = self._cache[newroi.toSlice()]
            newroi.start -= roi.start
            newroi.stop -= roi.start
            s = newroi.toSlice()
            result[s] = labels[chunk]

    def _toGlobal(self, labels):
        for l in np.unique(labels):
            if l == 0:
                continue
            if l not in self._globalLabels:
                with self._lock:
                    nextLabel = self._labelIterator.next()
                    self._globalLabels[l] = nextLabel
            labels[labels == l] = self._globalLabels[l]

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

    # returns an array of global labels in use by this chunk if 'mapping' is
    # False, a mapping of local labels to global labels otherwise
    def _getLabelsForChunk(self, chunkIndex, mapping=False):
        offset = self._globalLabelOffset[chunkIndex]
        numLabels = self._numLabels[chunkIndex]
        labels = np.arange(1, numLabels+1, dtype=_LABEL_TYPE) + offset
        if not mapping:
            return labels
        else:
            # we got 'numLabels' real labels, and one label '0', so our
            # output has to have numLabels+1 elements
            idx = np.arange(numLabels+1, dtype=_LABEL_TYPE)

            # real labels start at offset+1 and go up to (including)
            # numLabels+offset
            idx += offset

            # 0 always maps to 0!
            idx[0] = 0

            out = np.asarray(map(self._uf.findLabel, idx), dtype=_LABEL_TYPE)
            return out

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
