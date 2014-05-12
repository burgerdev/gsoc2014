#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
import vigra
import unittest

from numpy.testing import assert_array_equal, assert_array_almost_equal

from helpers import assertEquivalentLabeling, DirtyAssert
from lazycc import OpLazyCC as OpLabelVolume

from lazyflow.graph import Graph
from lazyflow.operator import Operator
from lazyflow.slot import InputSlot, OutputSlot
from lazyflow.rtype import SubRegion

from lazyflow.operators import OpArrayPiper, OpCompressedCache


class TestOpLazyCC(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skip("Takes too long with mockup UnionFind")
    def testSimpleUsage(self):
        vol = np.random.randint(255, size=(1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        op = OpLabelVolume(graph=Graph())
        op.Input.setValue(vol)
        op.ChunkShape.setValue((100, 10, 10))

        out = op.Output[...].wait()

        assert_array_equal(vol.shape, out.shape)

    def testCorrectLabeling(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Input.setValue(vol)
        op.ChunkShape.setValue((100, 10, 10))

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)

        assertEquivalentLabeling(vol, out)

    def testSingletonZ(self):
        vol = np.zeros((82, 70, 1), dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        blocks = np.zeros(vol.shape, dtype=np.uint8)
        blocks[30:50, 40:60, :] = 1
        blocks[60:70, 30:40, :] = 3
        blocks = vigra.taggedView(blocks, axistags='xyz')

        vol[blocks > 0] = 255

        op = OpLabelVolume(graph=Graph())
        op.ChunkShape.setValue((30, 25, 1))
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
        np.set_printoptions(threshold=np.nan, linewidth=200)
        print(out[..., 0])
        print(blocks[..., 0])
        assertEquivalentLabeling(blocks, out)

    def testLazyness(self):
        g = Graph()
        vol = np.asarray(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        vol = vigra.taggedView(vol, axistags='xy').withAxes(*'xyz')
        chunkShape = (3, 3, 1)
        
        opCount = OpExecuteCounter(graph=g)
        opCount.Input.setValue(vol)
        
        opCache = OpCompressedCache(graph=g)
        opCache.Input.connect(opCount.Output)
        opCache.BlockShape.setValue(chunkShape)
        
        op = OpLabelVolume(graph=g)
        op.Input.connect(opCache.Output)
        op.ChunkShape.setValue(chunkShape)
        
        out = op.Output[:3, :3].wait()
        n = 6
        assert opCount.numCalls <= n,\
            "Executed {} times (allowed: {})".format(opCount.numCalls,
                                                     n)
        

    def testConsistency(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLabelVolume(graph=Graph())
        op.Input.setValue(vol)
        op.ChunkShape.setValue((100, 10, 10))

        out1 = op.Output[:500, ...].wait()
        out2 = op.Output[500:, ...].wait()
        assert out1[0, 0, 0] != out2[499, 0, 0]

    def testSetDirty(self):
        g = Graph()
        vol = np.zeros((200, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLabelVolume(graph=g)
        op.ChunkShape.setValue((100, 20, 5))
        op.Input.setValue(vol)

        opCheck = DirtyAssert(graph=g)
        opCheck.Input.connect(op.Output)

        out = op.Output[:100, :20, :5].wait()

        roi = SubRegion(op.Input,
                        start=(0, 0, 0),
                        stop=(200, 100, 10))
        with self.assertRaises(PropagateDirtyCalled):
            op.Input.setDirty(roi)

        opCheck.Input.disconnect()


    @unittest.expectedFailure
    def testUnsupported(self):
        g = Graph()
        vol = np.zeros((50, 50))
        vol = vol.astype(np.int16)
        vol = vigra.taggedView(vol, axistags='xy')
        vol[:200, ...] = 1
        vol[800:, ...] = 1

        op = OpLabelVolume(graph=g)
        op.Method.setValue(self.method)
        with self.assertRaises(ValueError):
            op.Input.setValue(vol)

    @unittest.expectedFailure
    def testBackground(self):
        vol = np.zeros((1000, 100, 10))
        vol = vol.astype(np.uint8)
        vol = vigra.taggedView(vol, axistags='xyz')

        vol[20:40, 10:30, 2:4] = 1

        op = OpLabelVolume(graph=Graph())
        op.Method.setValue(self.method)
        op.Background.setValue(1)
        op.Input.setValue(vol)

        out = op.Output[...].wait()
        tags = op.Output.meta.getTaggedShape()
        out = vigra.taggedView(out, axistags="".join([s for s in tags]))

        assertEquivalentLabeling(1-vol, out)

        vol = vol.withAxes(*'xyzct')
        vol = np.concatenate(3*(vol,), axis=3)
        vol = np.concatenate(4*(vol,), axis=4)
        vol = vigra.taggedView(vol, axistags='xyzct')
        assert len(vol.shape) == 5
        assert vol.shape[3] == 3
        assert vol.shape[4] == 4

        op.Method.setValue(self.method)
        bg = np.asarray([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [1, 0, 0, 1]], dtype=np.uint8)
        bg = vigra.taggedView(bg, axistags='ct')
        assert len(bg.shape) == 2
        assert bg.shape[0] == 3
        assert bg.shape[1] == 4
        op.Background.setValue(bg)
        op.Input.setValue(vol)

        for c in range(bg.shape[0]):
            for t in range(bg.shape[1]):
                out = op.Output[..., c, t].wait()
                out = vigra.taggedView(out, axistags=op.Output.meta.axistags)
                if bg[c, t]:
                    assertEquivalentLabeling(1-vol[..., c, t], out.squeeze())
                else:
                    assertEquivalentLabeling(vol[..., c, t], out.squeeze())


class OpExecuteCounter(OpArrayPiper):

    def __init__(self, *args, **kwargs):
        self.numCalls = 0
        super(OpExecuteCounter, self).__init__(*args, **kwargs)

    def execute(self, slot, subindex, roi, result):
        self.numCalls += 1
        super(OpExecuteCounter, self).execute(slot, subindex, roi, result)


class DirtyAssert(Operator):
    Input = InputSlot()

    def propagateDirty(self, slot, subindex, roi):
        assert np.all(roi.start == 0)
        assert np.all(roi.stop == self.Input.meta.shape)
        raise PropagateDirtyCalled()


class PropagateDirtyCalled(Exception):
    pass


if __name__ == "__main__":
    vol = np.zeros((1000, 100, 10))
    vol[300:600, 40:70, 2:5] = 255
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')

    op = OpLabelVolume(graph=Graph())
    op.Input.setValue(vol)
    op.ChunkShape.setValue((100, 10, 10))

    out = op.Output[...].wait()
