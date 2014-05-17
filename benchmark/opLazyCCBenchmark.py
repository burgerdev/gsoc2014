#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc import OpLazyCC

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume

from timeit import timeit, repeat

import numpy as np
import vigra

if __name__ == "__main__":
    vol = np.zeros((1000, 100, 10))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')
    vol[:200, :10, ...] = 1
    vol[800:, ...] = 1
    
    op = OpLazyCC(graph=Graph())
    op.Input.setValue(vol)
    op.ChunkShape.setValue((100, 10, 10))

    res = timeit("out = op.Output[:50, :10, :].wait()",
                 setup="from __main__ import op", number=1)
    print("Needed {:.3f}ms for small region".format(res*1000))
    
    op2 = OpLabelVolume(graph=Graph())
    op2.Input.setValue(vol)
    res = timeit("out = op2.Output[:50, :10, :].wait()",
                 setup="from __main__ import op2", number=1)
    print("Compare to {:.3f}ms for OpLabelVolume small region (raw)".format(res*1000))
    res = timeit("out = op2.CachedOutput[:50, :10, :].wait()",
                 setup="from __main__ import op2", number=1)
    print("Compare to {:.3f}ms for OpLabelVolume small region (cached)".format(res*1000))

    op = OpLazyCC(graph=Graph())
    op.Input.setValue(vol)
    op.ChunkShape.setValue((100, 10, 10))

    res = timeit("out = op.Output[...].wait()",
                 setup="from __main__ import op", number=1)
    print("Needed {:.3f}ms for full volume, 100 chunks".format(res*1000))

    op = OpLazyCC(graph=Graph())
    op.Input.setValue(vol)
    op.ChunkShape.setValue((1000, 100, 10))

    res = timeit("out = op.Output[...].wait()",
                 setup="from __main__ import op", number=1)
    print("Needed {:.3f}ms for full volume, one chunk".format(res*1000))

    op2 = OpLabelVolume(graph=Graph())
    op2.Input.setValue(vol)
    res = timeit("out = op2.Output[...].wait()",
                 setup="from __main__ import op2", number=1)
    print("Compare to {:.3f}ms for OpLabelVolume full volume (raw)".format(res*1000))
    res = timeit("out = op2.CachedOutput[...].wait()",
                 setup="from __main__ import op2", number=1)
    print("Compare to {:.3f}ms for OpLabelVolume full volume (cached)".format(res*1000))
    
