#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc import OpLazyCC

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume

from timeit import timeit, repeat

import numpy as np
import vigra

def runSingleBenchmark(op, op2, vol, chunkShape):
    x, y, z = chunkShape
    op.Input.setValue(vol)
    op.ChunkShape.setValue(chunkShape)

    res = timeit("out = op.Output[:x, :y, :z].wait()",
                 setup="from __main__ import op, x, y, z", number=1)
    print("  Took {:.3f}ms for small region".format(res*1000))

    op2.Input.setValue(vol)
    op2.Input.setDirty(slice(None))
    res = timeit("out = op2.Output[:x, :y, :z].wait()",
                 setup="from __main__ import op2, x, y, z", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume small region (raw)".format(res*1000))
    res = timeit("out = op2.CachedOutput[:x, :y, :z].wait()",
                 setup="from __main__ import op2, x, y, z", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume small region (cached)".format(res*1000))
    #print(op2.CachedOutput[:10,:10, :1].wait().squeeze())

    op.Input.setValue(vol)
    op.ChunkShape.setValue(chunkShape)
    nChunks = np.prod(np.divide(vol.shape, chunkShape))

    res = timeit("out = op.Output[...].wait()",
                 setup="from __main__ import op", number=1)
    print("  Took {:.3f}ms for full volume, {} chunks".format(res*1000, nChunks))

    op.Input.setValue(vol)
    op.ChunkShape.setValue(vol.shape)

    res = timeit("out = op.Output[...].wait()",
                 setup="from __main__ import op", number=1)
    print("  Took {:.3f}ms for full volume, one chunk".format(res*1000))

    op2.Input.setValue(vol)
    op2.Input.setDirty(slice(None))
    res = timeit("out = op2.Output[...].wait()",
                 setup="from __main__ import op2", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume full volume (raw)".format(res*1000))
    res = timeit("out = op2.CachedOutput[...].wait()",
                 setup="from __main__ import op2", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume full volume (cached)".format(res*1000))

if __name__ == "__main__":
    op = OpLazyCC(graph=Graph())
    op2 = OpLabelVolume(graph=Graph())
    vol = np.zeros((1000, 100, 10))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')
    vol[:200, :10, ...] = 1
    vol[800:, ...] = 1
    chunkShape = (100, 10, 10)
    x, y, z = chunkShape
    print("Dense Labels")
    runSingleBenchmark(op, op2, vol, chunkShape)
    
    vol[:] = 0
    print("No Labels")
    runSingleBenchmark(op, op2, vol, chunkShape)

    vol[:] = np.random.randint(100, size=vol.shape) > 95 
    print("Sparse Labels")
    runSingleBenchmark(op, op2, vol, chunkShape)
