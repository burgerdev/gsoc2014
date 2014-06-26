#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring


from timeit import timeit, repeat
import resource
import psutil
import objgraph
from guppy import hpy

from lazycc import OpLazyCC

from lazyflow.graph import Graph
from lazyflow.operators import OpLabelVolume

import gc
from pprint import pprint

import numpy as np
import vigra

hp = hpy()


def profile(fun):
    def wrapped(*args, **kwargs):
        x = fun(*args, **kwargs)
        gc.collect()
        #objgraph.show_most_common_types()
        return x
    return wrapped


@profile
def runSingleBenchmark(op, vol, chunkShape):
    before = hp.heap()
    op.Input.setValue(vol)
    op.ChunkShape.setValue(chunkShape)
    op.Output[...].wait()
    after = hp.heap()
    leftover = after - before
    print(leftover)


if __name__ == "__main__":
    nRuns = 1
    chunkShape = (50, 50, 50)
    vol = np.zeros((200, 200, 200))
    vol = vol.astype(np.uint8)
    vol = vigra.taggedView(vol, axistags='xyz')
    vol[:60, :60, :60] = 1

    for i in range(nRuns):
        op = OpLazyCC(graph=Graph())
        #print("===========================")
        #print("Huge objects {}".format(i))
        runSingleBenchmark(op, vol, chunkShape)
        #print("===========================")

