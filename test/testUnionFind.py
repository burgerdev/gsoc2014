#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import unittest
import numpy as np
import vigra

from lazycc import mergeLabels, UnionFindArray, mapArray
from helpers import assertEquivalentLabeling


class TestUnionFind(unittest.TestCase):
    def setUp(self):
        tinyArray = [[10, 10, 0, 0],
                     [0, 10, 0, 10],
                     [10, 10, 0, 10]]
        self.tinyArray = np.asarray(tinyArray).astype(np.uint8)

    def testSimple(self):
        uf = UnionFindArray(np.uint8(3))
        for i in range(4):
            assert uf[i] == i
        uf.makeUnion(1,2)
        assert uf[1] == uf[2]
        for i in range(3):
            print("{}: {}".format(i, uf[i]))
        i = uf.makeContiguous()

    def testNumpyInit(self):
        for dt in (np.uint8, np.uint32, np.uint64):
            print("Trying to initialize UnionFindArray with data type {}".format(dt))
            x = np.arange(4, dtype=dt)
            uf = UnionFindArray(x[-1])
            uf.makeUnion(x[1], x[3])
            assert uf[1] == uf[3]

    def testLabelAlgorithm(self):
        x = self.tinyArray
        labels = np.zeros(x.shape, dtype=np.uint8)
        uf = UnionFindArray(labels)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] == 0:
                    continue
                if labels[i, j] == 0:
                    label = uf.makeNewLabel()
                    print("created label {}".format(label))
                    labels[i, j] = label
                # right
                if j < x.shape[1]-1 and x[i, j+1] == x[i, j]:
                    if labels[i, j+1] > 0:
                        uf.makeUnion(labels[i, j], labels[i, j+1])
                    else:
                        labels[i, j+1] = labels[i, j]

                # bottom
                if i < x.shape[0]-1 and x[i+1, j] == x[i, j]:
                    if labels[i+1, j] > 0:
                        uf.makeUnion(labels[i, j], labels[i+1, j])
                    else:
                        labels[i+1, j] = labels[i, j]

        mapArray(uf, labels)
        assert labels[-1, -1] != labels[0, 0]
        assert labels[-1, 0] == labels[0, 0]
