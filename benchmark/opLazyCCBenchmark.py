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
    print("  Took {:.3f}ms for one chunk".format(res*1000))

    # benchmark calculation of one chunk
    op2.Input.setValue(vol)
    op2.Input.setDirty(slice(None))
    res = timeit("out = op2.Output[:x, :y, :z].wait()",
                 setup="from __main__ import op2, x, y, z", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume one chunk (raw)".format(res*1000))
    res = timeit("out = op2.CachedOutput[:x, :y, :z].wait()",
                 setup="from __main__ import op2, x, y, z", number=1)
    print("  Compare to {:.3f}ms for OpLabelVolume one chunk (cached)".format(res*1000))

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
    vol = np.zeros((200, 200, 200))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')
    vol[:60, :60, :60] = 1
    chunkShape = (50, 50, 50)
    x, y, z = chunkShape
    print("===========================")
    print("Huge objects")
    runSingleBenchmark(op, op2, vol, chunkShape)
    print("===========================")

    vol[:] = 0
    print("No Objects")
    runSingleBenchmark(op, op2, vol, chunkShape)
    print("===========================")

    # want to have few objects on boundaries (250*2 elements on boundary)
    # chance of 1/4 of an pobject to ly on boundary
    vol[:] = np.random.randint(2000, size=vol.shape) == 0
    print("Sparse Objects")
    runSingleBenchmark(op, op2, vol, chunkShape)
    print("===========================")
