#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazyflow.operator import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from _merge import mergeLabels

## 3d data only, xyz
class OpLazyCC(Operator):
    Input = InputSlot()
    ChunkShape = InputSlot()
    Output = OutputSlot()
    _FakeOutput = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super(OpLazyCC, self).__init__(*args, **kwargs)
        
        self._cache = OpCompressedCache(parent=self)
        self._cache.Input.connect(self._FakeOutput)
        
    
    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Output.meta.dtype = np.uint32
        self._FakeOutput.meta.assignFrom(self.Output.meta)
        
        # chunk array shape calculation
        shape = self.Input.meta.shape
        chunkShape = self.ChunkShape.value
        f = lambda i: shape[i]//chunkShape[i] + (1 if shape[i]%chunkShape[i] else 0)
        self._chunkArrayShape = tuple(map(f,range(3)))
        self._chunkShape = np.asarray(chunkShape, dtype=np.int)
        
        # keep track of number of labels in chunk (-1 == not labeled yet)
        #FIXME need int64 here, but we are screwed anyway if the number of 
        # labels in a single chunk exceeds 2**31
        self._numLabels = -np.ones(self._chunkArrayShape, dtype=np.int32)
        
        # keep track of what has been finalized
        self._isFinal = np.zeros(self._chunkArrayShape, dtype=np.bool)
        
        # offset (global labels - local labels) per chunk
        self._globalLabelOffset = -np.ones(self._chunkArrayShape, dtype=np.int32)
        
        
    ## grow the requested region such that all labels inside that region are final
    # @param chunkIndex the index of the chunk to finalize
    # @param labels the labels that need to be finalized
    def _finalizeChunk(self, chunkIndex, labels):
        #FIXME lock this in case some other thread wants to finalize the same chunk
        if self._isFinal[chunkIndex]:
            return
        
        # get a list of neighbours that have to be checked
        neighbours = self._generateNeighbours(chunkIndex)
        
        # label this chunk first
        self._label(chunkIndex)
        
        for otherChunk in neighbours:
            self._label(otherChunk)
            self._merge(chunkIndex, otherChunk)
            
            adjacentLabels = self._getAdjacentLabels(chunkIndex, otherChunk)
            adjacentlabels = map(lambda s: s[1], adjacentLabels)
            
            self._finalize(otherChunk, adjacentLabels)

    ## label a chunk and store information
    def _label(self, chunkIndex):
        #FIXME prevent other threads from labeling this block
        if self._numLabels[chunkIndex] >= 0:
            # this chunk is already labeled
            return
        
        # get the raw data
        roi = self._chunkIndexToRoi(chunkIndex)
        inputChunk = self.Input.get(roi).wait()
        
        # label the raw data
        labeled = vigra.labelVolumeWithBackground(temp)
        
        # store the labeled data in cache
        self._cache.setInSlot(self._cache.Input, (), roi, labeled)
        
        # update the labeling information
        numLabels = labeled.max()  # we ignore 0 here
        self._numLabels[chunkIndex] = numLabels
        if numLabels > 0:
            #FIXME critical section here
            # get 1 label that determines the offset
            offset = self._uf.makeNewLabel()
            # the offset is such that label 1 in the local chunk maps to
            # 'offset' in the global context
            self._globalLabelOffset[chunkIndex] = offset - 1
            
            # get n-1 more labels
            for i in range(numLabels-1):
                self._uf.makeNewLabel()
    
    ## merge chunks with callback that updates adjacency graph
    def _merge(self, chunkA, chunkB):
        #TODO
        pass

    def _chunkIndexToRoi(self, index):
        shape = self.Input.meta.shape
        stop = self._chunkShape
        stop = np.where(stop > shape, shape, stop)
        start = stop * 0
        roi = SubRegion(self.Input,
                        start=tuple(start), stop=tuple(stop))
        return roi

    def _getAdjacentLabels(self, indexA, indexB):
        return []

    def _generateNeighbours(self, chunkIndex):
        n = []
        idx = np.asarray(chunkIndex, dtype=np.int)
        for i in len(chunkIndex):
            if idx[i] > 0:
                new = idx.copy()
                new[i] -= 1
                n.append(tuple(new))
            if idx[i]+1 < self._chunkArrayShape[i]:
                new = idx.copy()
                new[i] += 1
                n.append(tuple(new))
        return n
            
