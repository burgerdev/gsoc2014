#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
import vigra
import logging

from collections import defaultdict
from functools import partial, wraps
#from itertools import count as InfiniteLabelIterator
from _tools import InfiniteLabelIterator

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpCompressedCache, OpReorderAxes
from lazyflow.request import Request, RequestPool
from lazyflow.request import RequestLock as ReqLock
# the lazyflow lock seems to have deadlock issues sometimes
from threading import Lock as HardLock
from threading import Condition

from _mockup import UnionFindArray
#from lazycc import UnionFindArray

# logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

_LABEL_TYPE = np.uint32


def threadsafe(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class _LabelManager(object):

    def __init__(self):
        self._lock = Condition()
        self._managedLabels = defaultdict(dict)
        self._iterator = InfiniteLabelIterator(1, dtype=int)
        self._registered = set()

    # call before doing anything
    @threadsafe
    def register(self):
        n = self._iterator.next()
        self._registered.add(n)
        return n

    # call when done with everything
    @threadsafe
    def unregister(self, n):
        self._registered.remove(n)
        self._lock.notify_all()

    # call to wait for other processes
    @threadsafe
    def waitFor(self, others):
        others = set(others)
        remaining = others & self._registered
        while len(remaining) > 0:
            self._lock.wait()
            remaining &= self._registered

    # get a list of labels that _really_ need to be globalized by you
    @threadsafe
    def checkoutLabels(self, chunkIndex, labels, n):
        others = set()
        d = self._managedLabels[chunkIndex]
        for otherProcess, otherLabels in d.iteritems():
            inters = np.intersect1d(labels, otherLabels)
            if inters.size > 0:
                labels = np.setdiff1d(labels, inters)
                others.add(otherProcess)
        if labels.size > 0:
            d[n] = labels
        return labels, others


# locking decorator that locks per chunk
def _chunksynchronized(method):
    @wraps(method)
    def synchronizedmethod(self, chunkIndex, *args, **kwargs):
        with self._chunk_locks[chunkIndex]:
            return method(self, chunkIndex, *args, **kwargs)
    return synchronizedmethod


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
#       output, then, for every global label i<j, i also appears in the output.
#       The actual implementation is hidden in self.globalToFinal().
#       aka 'final'
#
class OpLazyCC(Operator):

    # input data (usually segmented), in 'txyzc' order
    Input = InputSlot()

    # the spatial shape of one chunk, in 'xyz' order
    ChunkShape = InputSlot()

    # the labeled output, internally cached
    Output = OutputSlot()

    ### INTERNALS -- DO NOT USE ###
    _Input = OutputSlot()
    _Output = OutputSlot()

    def __init__(self, *args, **kwargs):
        super(OpLazyCC, self).__init__(*args, **kwargs)
        self._lock = HardLock()

        # reordering operators - we want to handle txyzc inside this operator
        self._opIn = OpReorderAxes(parent=self)
        self._opIn.AxisOrder.setValue('txyzc')
        self._opIn.Input.connect(self.Input)
        self._Input.connect(self._opIn.Output)

        self._opOut = OpReorderAxes(parent=self)
        self._opOut.Input.connect(self._Output)
        self.Output.connect(self._opOut.Output)

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = _LABEL_TYPE
        self._Output.meta.assignFrom(self._Input.meta)
        self._Output.meta.dtype = _LABEL_TYPE
        assert self.Input.meta.dtype in [np.uint8, np.uint32, np.uint64],\
            "Cannot label data type {}".format(self.Input.meta.dtype)

        self._setDefaultInternals()

        # go back to original order
        self._opOut.AxisOrder.setValue(self.Input.meta.getAxisKeys())

    def execute(self, slot, subindex, roi, result):
        if slot is self._Output:
            othersToWaitFor = set()
            chunks = self._roiToChunkIndex(roi)
            for chunk in chunks:
                othersToWaitFor |= self.growRegion(chunk)

            self._manager.waitFor(othersToWaitFor)
            self._mapArray(roi, result)
        else:
            raise ValueError("Request to invalid slot {}".format(str(slot)))

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
    def growRegion(self, chunkIndex):
        ticket = self._manager.register()
        othersToWaitFor = set()

        # we want to finalize every label in our first chunk
        localLabels = np.arange(1, self._numIndices[chunkIndex]+1)
        localLabels = localLabels.astype(_LABEL_TYPE)
        chunksToProcess = {chunkIndex: localLabels}

        while chunksToProcess:
            currentChunk, localLabels = chunksToProcess.popitem()

            # label this chunk
            self._label(chunkIndex)

            # get the labels in use by this chunk
            localLabels = np.arange(1, self._numIndices[currentChunk]+1)
            localLabels = localLabels.astype(_LABEL_TYPE)

            # tell the label manager that we are about to finalize some labels
            actualLabels, others = self._manager.checkoutLabels(currentChunk,
                                                                localLabels,
                                                                ticket)
            othersToWaitFor |= others

            # now we have got a list of local labels for this chunk, which no
            # other process is going to finalize

            # start merging adjacent regions
            otherChunks = self._generateNeighbours(currentChunk)
            for other in otherChunks:
                self._label(other)
                a, b = self._orderPair(currentChunk, other)
                me = 0 if a == chunkIndex else 1
                res = self._merge(a, b)
                myLabels, otherLabels = res[me], res[1-me]

                # determine which objects from this chunk continue in the
                # neighbouring chunk
                extendingLabels = [b for a, b in zip(myLabels, otherLabels)
                                   if a in actualLabels]
                extendingLabels = np.unique(extendingLabels
                                            ).astype(_LABEL_TYPE)

                # add the neighbour to our processing queue only if it actually
                # shares objects
                if extendingLabels.size > 0:
                    if other in chunksToProcess:
                        extendingLabels = np.union1d(chunksToProcess[other],
                                                     extendingLabels)
                    chunksToProcess[other] = extendingLabels

        self._manager.unregister(ticket)
        return othersToWaitFor

    # label a chunk and store information
    @_chunksynchronized
    def _label(self, chunkIndex):
        if self._numIndices[chunkIndex] >= 0:
            # this chunk is already labeled
            return

        # get the raw data
        roi = self._chunkIndexToRoi(chunkIndex)
        inputChunk = self._Input.get(roi).wait()
        inputChunk = vigra.taggedView(inputChunk, axistags='txyzc')
        inputChunk = inputChunk.withAxes(*'xyz')

        # label the raw data
        labeled = vigra.analysis.labelVolumeWithBackground(inputChunk)
        labeled = vigra.taggedView(labeled, axistags='xyz').withAxes(*'txyzc')
        del inputChunk
        #TODO this could be more efficiently combined with merging

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
    # the chunks have to be ordered lexicographically, e.g. by self._orderPair
    @_chunksynchronized
    def _merge(self, chunkA, chunkB):
        if chunkB in self._mergeMap[chunkA]:
            return (np.zeros((0,), dtype=_LABEL_TYPE),)*2
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
            return (np.zeros((0,), dtype=_LABEL_TYPE),)*2

        # check if the labels do actually belong to the same component
        hyperplane_a = self._Input[hyperplane_index_a].wait()
        hyperplane_b = self._Input[hyperplane_index_b].wait()
        adjacent_bool_inds = np.logical_and(adjacent_bool_inds,
                                            hyperplane_a == hyperplane_b)

        # union find manipulations are critical
        with self._lock:
            map_a = self.localToGlobal(chunkA, mapping=True)
            map_b = self.localToGlobal(chunkB, mapping=True)
            labels_a = map_a[label_hyperplane_a[adjacent_bool_inds]]
            labels_b = map_b[label_hyperplane_b[adjacent_bool_inds]]
            for a, b in zip(labels_a, labels_b):
                self._uf.makeUnion(a, b)
        correspondingLabelsA = label_hyperplane_a[adjacent_bool_inds]
        correspondingLabelsB = label_hyperplane_b[adjacent_bool_inds]
        return correspondingLabelsA, correspondingLabelsB

    # get a rectangular region with final global labels
    # @param roi region of interest
    # @param result array of shape roi.stop - roi.start, will be filled
    def _mapArray(self, roi, result, global_labels=True):
        # TODO perhaps with pixeloperator?
        assert np.all(roi.stop - roi.start == result.shape)
        indices = self._roiToChunkIndex(roi)
        for idx in indices:
            newroi = self._chunkIndexToRoi(idx)
            newroi.stop = np.minimum(newroi.stop, roi.stop)
            newroi.start = np.maximum(newroi.start, roi.start)
            self._mapChunk(idx)
            chunk = self._cache[newroi.toSlice()]
            newroi.start -= roi.start
            newroi.stop -= roi.start
            s = newroi.toSlice()
            result[s] = chunk

    @_chunksynchronized
    def _mapChunk(self, chunkIndex):
        if self._isFinal[chunkIndex]:
            return

        newroi = self._chunkIndexToRoi(chunkIndex)
        s = newroi.toSlice()
        chunk = self._cache[s]
        labels = self.localToGlobal(chunkIndex, mapping=True)
        self.globalToFinal(chunkIndex[0], chunkIndex[4], labels)
        self._cache[s] = labels[chunk]

        self._isFinal[chunkIndex] = True

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
    # after calling this function, the labels passed in may not be used with
    # UnionFind.makeUnion any more!
    @threadsafe
    def globalToFinal(self, t, c, labels):
        d = self._globalToFinal[(t, c)]
        labeler = self._labelIterators[(t, c)]
        for k in np.unique(labels):
            l = self._uf.findIndex(k)
            if l == 0:
                continue

            # adding a global label is critical
            if l not in d:
                nextLabel = labeler.next()
                d[l] = nextLabel
            labels[labels == l] = d[l]

    ##########################################################################
    ##################### HELPER METHODS #####################################
    ##########################################################################

    # create roi object from chunk index
    def _chunkIndexToRoi(self, index):
        shape = self._shape
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
        #TODO do this smarter, perhaps?
        for t in range(start_cs[0], stop_cs[0]):
            for x in range(start_cs[1], stop_cs[1]):
                for y in range(start_cs[2], stop_cs[2]):
                    for z in range(start_cs[3], stop_cs[3]):
                        for c in range(start_cs[4], stop_cs[4]):
                            chunks.append((t, x, y, z, c))
        return chunks

    # compute the adjacent hyperplanes of two chunks (1 pix wide)
    # @return 2-tuple of roi's for the respective chunk
    def _chunkIndexToHyperplane(self, chunkA, chunkB):
        rev = False
        assert chunkA[0] == chunkB[0] and chunkA[4] == chunkB[4],\
            "these chunks are not spatially adjacent"

        # just iterate over spatial axes
        for i in range(1, 4):
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
        # only spatial neighbours are considered
        for i in range(1, 4):
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
        #TODO change here when removing OpReorder
        shape = self._Input.meta.shape
        chunkShape = (1,) + self.ChunkShape.value + (1,)
        assert len(shape) == len(chunkShape),\
            "Encountered an invalid chunkShape"
        f = lambda i: shape[i]//chunkShape[i] + (1 if shape[i] % chunkShape[i]
                                                 else 0)
        self._chunkArrayShape = tuple(map(f, range(len(shape))))
        self._chunkShape = np.asarray(chunkShape, dtype=np.int)
        self._shape = shape

        # manager object
        self._manager = _LabelManager()

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
        gen = partial(InfiniteLabelIterator, 1, dtype=_LABEL_TYPE)
        self._labelIterators = defaultdict(gen)
        self._globalToFinal = defaultdict(dict)
        self._isFinal = np.zeros(self._chunkArrayShape, dtype=np.bool)

        ### algorithmic ###

        # keep track of merged regions
        self._mergeMap = defaultdict(list)

        # locks that keep threads from changing a specific chunk
        self._chunk_locks = defaultdict(HardLock)

    # order a pair of chunk indices lexicographically
    # (ret[0] is top-left-in-front-of of ret[1])
    @staticmethod
    def _orderPair(tupA, tupB):
        for a, b in zip(tupA, tupB):
            if a < b:
                return tupA, tupB
            if a > b:
                return tupB, tupA
        raise ValueError("tupA={} and tupB={} are the same".format(tupA, tupB))
        return tupA, tupB
