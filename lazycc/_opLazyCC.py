#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazyflow.operator import Operator, InputSlot, OutputSlot

from _merge import mergeLabels

## 3d data only, xyz
class OpLazyCC(lazyflow.operator.Operator):
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
        f = lambda i: shape[i]//chunkShape[i] + (1 shape[i]%chunkShape[i] else 0)
        self._chunkArrayShape = tuple(map(f,range(3)))
        
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
        if self._isFinal[chunkIndex]:
            return
        
        neighbours = self._generateNeighbours(chunkIndex)
        
        for otherChunk in neighbours:
            self._label(otherChunk)
            self._merge(otherChunk, chunkIndex)
            
            adjacentLabels = self._getAdjacentLabels(chunkIndex, otherChunk)
            adjacentlabels = map(lambda s: s[1], adjacentLabels)
            
            self._finalize(otherChunk, adjacentLabels)

    ## label a chunk and store information
    def _label(self, chunkIndex):
        if self._numLabels[chunkIndex] >= 0:
            return
        roi = self._chunkIndexToRoi(chunkIndex)
        temp = self.Input.get(roi).wait()
        labeled = vigra.labelVolumeWithBackground(temp)
        self._cache.setInSlot(self._cache.Input, (), roi, labeled)
        numLabels = labeled.max()  # we ignore 0 here
        self._numLabels[chunkIndex] = numLabels
        if numLabels > 0:
            offset = self._uf.makeNewLabel()
            self._globalLabelOffset[chunkIndex] = offset  # FIXME +1??
            
        
