#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc import OpLazyCC

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume

from timeit import timeit, repeat

import numpy as np
import vigra

import cProfile, pstats, StringIO

if __name__ == "__main__":
    vol = np.zeros((200, 200, 200))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')
    vol[:200, :10, ...] = 1
    vol[800:, ...] = 1

    op = OpLazyCC(graph=Graph())
    op.Input.setValue(vol)
    op.ChunkShape.setValue((50, 50, 50))

    pr = cProfile.Profile()
    pr.enable()
    # start calculation #
    out = op.Output[...].wait()

    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    # ps.strip_dirs().print_callers('acquire')
    # ps.strip_dirs().print_callers('\(wait\)')
    print(s.getvalue())

