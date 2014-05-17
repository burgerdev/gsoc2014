#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
import vigra
import logging

from collections import defaultdict
from threading import RLock
from functools import partial, wraps

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
from lazyflow.operators import OpCompressedCache
from lazyflow.request import Request, RequestPool

from lazycc import mergeLabels
from lazycc import UnionFindArray

#logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

_LABEL_TYPE = np.uint32




## 3d data only, xyz
class OpLazyCC(Operator):
    Input = InputSlot()
    ChunkShape = InputSlot()
    Output = OutputSlot()

    _FakeOutput = OutputSlot()


    def chunksynchronized(method):
        @wraps(method)
        def synchronizedmethod(self, chunkIndex, *args, **kwargs):
            with self._chunk_locks[chunkIndex]:
                return method(self, chunkIndex, *args, **kwargs)
        return synchronizedmethod
    
    def synchronized(method):
        @wraps(method)
        def synchronizedmethod(self, *args, **kwargs):
            with self._lock:
                return method(self, *args, **kwargs)
        return synchronizedmethod

    def __init__(self, *args, **kwargs):
        super(OpLazyCC, self).__init__(*args, **kwargs)

        self._cache = OpCompressedCache(parent=self)
        self._cache.Input.connect(self._FakeOutput)
        self._cache.BlockShape.connect(self.ChunkShape)

        self._lock = RLock()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = _LABEL_TYPE
        self._FakeOutput.meta.assignFrom(self.Output.meta)
        assert self.Input.meta.dtype in [np.uint8, np.uint32, np.uint64]

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
        self._numLabels = -np.ones(self._chunkArrayShape, dtype=np.int64)
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

    def execute(self, slot, subindex, roi, result):
        assert slot is not self._FakeOutput, "request to _FakeOutput: {}".format(roi)

        chunks = self._roiToChunkIndex(roi)
        for chunk in chunks:
            self._finalize(chunk)

        self._mapArray(roi, result)

    def propagateDirty(self, slot, subindex, roi):
        # TODO: this is already correct, but may be over-zealous
        # we would have to label each chunk that was set dirty and check
        # for changed labels. THerefore, we would have to check if the
        # dirty region is 'small enough', etc etc.
        self.Output.setDirty(slice(None))

    ## grow the requested region such that all labels inside that region are
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

        pool = RequestPool()
        for otherChunk in neighbours:
            pool.add(Request(partial(processNeighbour, otherChunk)))
        pool.wait()
        pool.clean()

        with self._lock:
            if labels is None:
                labels = self._getLabelsForChunk(chunkIndex)
            else:
                # update the labels to the latest state
                labels = map(self._uf.find, labels)
            if self._finalizedLabels[chunkIndex] is None:
                self._finalizedLabels[chunkIndex] = np.zeros((0,),
                                                             dtype=_LABEL_TYPE)
            otherLabels = map(lambda x: self._getLabelsForChunk(x), neighbours)

            # let the others know that we are finalizing this chunk
            finalized = map(self._uf.find, self._finalizedLabels[chunkIndex])
            now_finalized = np.union1d(finalized, labels).astype(_LABEL_TYPE)
            self._finalizedLabels[chunkIndex] = now_finalized
        labels = np.setdiff1d(labels, finalized)

        for i, l in zip(neighbours, otherLabels):
            d = np.intersect1d(labels, l, assume_unique=True)
            # don't go to finalized neighbours
            if self._finalizedLabels[i] is not None:
                finalized = map(self._uf.find, self._finalizedLabels[i])
                d = np.setdiff1d(d, finalized)
            d = d.astype(_LABEL_TYPE)
            if len(d) > 0:
                logger.debug("Going from {} to {} because of labels {}".format(chunkIndex, i, d))
                # start DFS recursion
                self._finalize(i, labels=d)

    ## label a chunk and store information
    @chunksynchronized
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
        #logger.info("Done labeling")
        del inputChunk

        # store the labeled data in cache
        #logger.info("Caching {} into {} ...".format(chunkIndex, roi))
        self._cache.setInSlot(self._cache.Input, (), roi, labeled)
        #logger.info("Done caching")

        # update the labeling information
        numLabels = labeled.max()  # we ignore 0 here
        self._numLabels[chunkIndex] = numLabels
        if numLabels > 0:
            with self._lock:
                # get 1 label that determines the offset
                offset = self._uf.makeNewLabel()
                # the offset is such that label 1 in the local chunk maps to
                # 'offset' in the global context
                self._globalLabelOffset[chunkIndex] = offset - 1

                # get n-1 more labels
                for i in range(numLabels-1):
                    self._uf.makeNewLabel()

    # merge chunks
    @chunksynchronized
    def _merge(self, chunkA, chunkB):
        if chunkB in self._mergeMap[chunkA]:
            return
        logger.info("Merging {} {} ...".format(chunkA, chunkB))

        hyperplane_index_a, hyperplane_index_b = \
            self._chunkIndexToHyperplane(chunkA, chunkB)

        hyperplane_a = self.Input[hyperplane_index_a].wait()
        hyperplane_b = self.Input[hyperplane_index_b].wait()
        label_hyperplane_a = self._cache.Output[hyperplane_index_a].wait()
        label_hyperplane_b = self._cache.Output[hyperplane_index_b].wait()

        map_a = self._getLabelsForChunk(chunkA, mapping=True)
        map_b = self._getLabelsForChunk(chunkB, mapping=True)

        # union find manipulations are critical
        with self._lock:
            mergeLabels(hyperplane_a, hyperplane_b,
                        label_hyperplane_a.astype(_LABEL_TYPE),
                        label_hyperplane_b.astype(_LABEL_TYPE),
                        map_a, map_b, self._uf)
        self._mergeMap[chunkA].append(chunkB)

    @synchronized
    def _mapArray(self, roi, result):
        # TODO perhaps with pixeloperator?
        assert np.all(roi.stop -roi.start == result.shape)
        indices = self._roiToChunkIndex(roi)
        for idx in indices:
            newroi = self._chunkIndexToRoi(idx)
            newroi.stop = np.minimum(newroi.stop, roi.stop)
            newroi.start = np.maximum(newroi.start, roi.start)
            labels = self._getLabelsForChunk(idx, mapping=True)
            chunk = self._cache.Output.get(newroi).wait()
            newroi.start -= roi.start
            newroi.stop -= roi.start
            s = newroi.toSlice()
            result[s] = labels[chunk]

    def _chunkIndexToRoi(self, index):
        shape = self.Input.meta.shape
        start = self._chunkShape * np.asarray(index)
        stop = self._chunkShape * (np.asarray(index) + 1)
        stop = np.where(stop > shape, shape, stop)
        roi = SubRegion(self.Input,
                        start=tuple(start), stop=tuple(stop))
        return roi

    def _roiToChunkIndex(self, roi):
        cs = self._chunkShape
        start = np.asarray(roi.start)
        stop = np.asarray(roi.stop)
        start_cs = start / cs
        stop_cs = stop / cs
        stop_mod = stop % cs
        stop_cs += np.where(stop_mod, 1, 0)
        chunks = []
        for x in range(start_cs[0], stop_cs[0]):
            for y in range(start_cs[1], stop_cs[1]):
                for z in range(start_cs[2], stop_cs[2]):
                    chunks.append((x, y, z))
        return chunks

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
            return roiB.toSlice(), roiA.toSlice()
        else:
            return roiA.toSlice(), roiB.toSlice()

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

    # returns an array of labels in use if mapping is False, a mapping
    # of local labels to global labels otherwise
    @synchronized
    def _getLabelsForChunk(self, chunkIndex, mapping=False):
        offset = self._globalLabelOffset[chunkIndex]
        numLabels = self._numLabels[chunkIndex]
        labels = np.arange(1, numLabels+1, dtype=_LABEL_TYPE) + offset
        if not mapping:
            return np.unique(map(self._uf.find, labels)).astype(_LABEL_TYPE)
        else:
            # we got 'numLabels' real labels, and one label '0', so our
            # output has to have numLabels+1 elements
            idx = np.arange(numLabels+1, dtype=_LABEL_TYPE)
            
            # real labels start at offset+1 and go up to (including)
            # numLabels+offset
            idx += offset
            
            # 0 always maps to 0!
            idx[0] = 0
            
            out = np.asarray(map(self._uf.find, idx), dtype=_LABEL_TYPE)
            return out

    @staticmethod
    def _orderPair(tupA, tupB):
        for a, b in zip(tupA, tupB):
            if a < b:
                return tupA, tupB
            if a > b:
                return tupB, tupA
        logger.warn("tupA and tupB are the same")
        return tupA, tupB
