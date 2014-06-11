#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import resource

import unittest
import numpy as np
import vigra

from lazycc import mergeLabels, UnionFindArray
from helpers import assertEquivalentLabeling


class TestMergeLabels(unittest.TestCase):
    def setUp(self):
        pass

    def testTheOneAndOnly(self):
        left = np.asarray([0, 0, 1, 3], dtype=np.uint8)[:, np.newaxis, np.newaxis]
        right = np.asarray([0, 0, 2, 3], dtype=np.uint8)[:, np.newaxis, np.newaxis]
        llabels = np.asarray([0, 1, 2, 3], dtype=np.uint32)[:, np.newaxis, np.newaxis]
        rlabels = np.asarray([0, 1, 2, 3], dtype=np.uint32)[:, np.newaxis, np.newaxis]
        lmap = llabels.copy().squeeze()
        rmap = rlabels.copy().squeeze() + 4
        rmap[0] = 0
        uf = UnionFindArray(rmap)
        mergeLabels(left, right, llabels, rlabels, lmap, rmap, uf)

        d = dict([(i, uf.findLabel(i)) for i in range(uf.nextFreeIndex())])
        print(d)

        assert uf.findLabel(5) == uf.findLabel(1)
        assert uf.findLabel(7) == uf.findLabel(3)
        assert uf.findLabel(6) != uf.findLabel(2)

    def testVariousArrays(self):
        for d in range(1, 3):
            for pt in (np.uint8, np.uint32, np.uint64, np.float32):
                #for lt in (np.uint8, np.uint32, np.uint64):
                for lt in (np.uint32,):
                    print("{}-dim, pixel type: {}, label type: {}".format(d, pt, lt))
                    shape = (5,)*d

                    maxInt = 255
                    #maxInt = np.iinfo(pt).max/2
                    x = np.random.randint(maxInt, size=shape).astype(pt)
                    y = np.random.randint(maxInt, size=shape).astype(pt)

                    maxInt = 122
                    #maxInt = np.iinfo(pt).max/2
                    xl = np.random.randint(maxInt, size=shape).astype(lt)//3
                    yl = np.random.randint(maxInt, size=shape).astype(lt)//3

                    m = max(xl.flat)
                    n = max(yl.flat)
                    xm = np.arange(m+1, dtype=lt)
                    ym = np.arange(n+1, m+n+1, dtype=lt)
                    print("Max label: {}".format(m+n))

                    uf = UnionFindArray(ym)

                    mergeLabels(x, y, xl, yl, xm, ym, uf)

