#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc._merge import mergeLabels as pyMergeLabels
from lazycc._lazycc_cxx import mergeLabels as cMergeLabels
from lazycc._lazycc_cxx import mergeLabelsRaw as cMergeLabelsRaw
from lazycc import UnionFindArray

from timeit import timeit, repeat

import numpy as np
import vigra


if __name__ == "__main__":
    N = 200
    shape = (64, 64)

    labelType = np.uint32

    # arrays must be in vigra order for cMergeLabelsRaw to work
    left = np.zeros(shape, dtype=np.uint8).transpose()
    right = np.zeros(shape, dtype=np.uint8).transpose()

    left[:] = np.random.randint(255, size=shape).astype(np.uint8)
    right[:] = left
    #right[:] = np.random.randint(255, size=shape).astype(np.uint8)

    labels_left = np.zeros(left.shape, dtype=labelType).transpose()
    labels_right = np.zeros(right.shape, dtype=labelType).transpose()

    #res = timeit('labelImage(right)', setup='from vigra.analysis import labelImage; from __main__ import right', number=N)
    #print(res)
    vigra.analysis.labelImage(left, out=labels_left)
    vigra.analysis.labelImage(right, out=labels_right)

    max_left = np.max(labels_left)
    max_right = np.max(labels_right)

    map_left = np.arange(max_left+1, dtype=labelType)
    map_right = np.arange(max_right+1, dtype=labelType) + max_left
    map_right[0] = 0

    setup = "from __main__ import {}, "\
            "left, right, labels_left, labels_right, "\
            "map_left, map_right, max_left, max_right, UnionFindArray;"\
            "uf = UnionFindArray(map_right)"
    cmd = "{}(left, right, labels_left, labels_right, "\
          "map_left, map_right, uf)"

    for impl in ["pyMergeLabels", "cMergeLabels", "cMergeLabelsRaw"]:
        print("{} for shape {}:".format(cmd.format(impl), left.shape))
        res = repeat(cmd.format(impl), setup=setup.format(impl),
                     repeat=3, number=N)
        print("    " + " ".join(["{:.3f}s".format(r) for r in res]))

