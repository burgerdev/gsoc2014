#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from lazycc._merge import mergeLabels as pyMergeLabels
from lazycc._lazycc_cxx import mergeLabels as cMergeLabels
from lazycc._lazycc_cxx import mergeLabelsInspect as cMergeLabelsInspect
from lazycc import UnionFindArray

from timeit import timeit, repeat

import numpy as np
import vigra


if __name__ == "__main__":

    labelType = np.uint32

    left = np.random.randint(255, size=(1, 64, 64)).astype(np.uint8)
    right = np.random.randint(255, size=(1, 64, 64)).astype(np.uint8)

    labels_left = np.zeros(left.shape, dtype=labelType)
    labels_right = np.zeros(right.shape, dtype=labelType)

    res = timeit('labelVolume(right)', setup='from vigra.analysis import labelVolume;from __main__ import right', number=50)
    print(res)
    vigra.analysis.labelVolume(left, out=labels_left)
    vigra.analysis.labelVolume(right, out=labels_right)

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

    for impl in ["pyMergeLabels", "cMergeLabels", "cMergeLabelsInspect"]:
        print("{} for shape {}:".format(cmd.format(impl), left.shape))
        res = repeat(cmd.format(impl), setup=setup.format(impl),
                     repeat=1, number=50)
        print("    " + " ".join(["{:.3f}s".format(r) for r in res]))

