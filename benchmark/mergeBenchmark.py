#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc._merge import mergeLabels as pyMergeLabels
from lazycc._lazycc_cxx import mergeLabels as cMergeLabels
from lazycc import UnionFindArray

from timeit import timeit, repeat

import numpy as np
import vigra


if __name__ == "__main__":
    import mpi4py

    x = np.random.randint(255, size=(100, 100, 100)).astype(np.uint8)
    y = np.random.randint(255, size=(100, 100, 100)).astype(np.uint8)
    
    uf = UnionFindArray(np.uint32(1))
    
    labels_x = vigra.analysis.labelVolume(x)
    labels_y = vigra.analysis.labelVolume(y)
    
    m = np.max(labels_x)
    n = np.max(labels_y)
    
    [uf.makeNewLabel() for i in range(m+n)]
    
    xmap = np.arange(m+1, dtype=np.uint32)
    ymap = np.arange(n+1, dtype=np.uint32) + m
    ymap[0] = 0
    
    setup = "from __main__ import {}, x, y, labels_x, labels_y, "\
            "xmap, ymap, m, n, UnionFindArray;"\
            "uf = UnionFindArray(labels_y + m)"
    cmd = "{}(x, y, labels_x, labels_y, xmap, ymap, uf)"
    
    for impl in ["pyMergeLabels", "cMergeLabels"]:
        print("{} for shape {}:".format(cmd.format(impl), x.shape))
        res = repeat(cmd.format(impl), setup=setup.format(impl),
                     repeat=10, number=1)
        print("    " + " ".join(["{:.3f}s".format(r) for r in res]))

