#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import unittest
import numpy as np
import vigra

from lazycc import *
from helpers import assertEquivalentLabeling


def mapArray(uf, x):
    n = uf.nextFreeIndex()
    T = type(uf)
    if T == UnionFindUInt8:
        dt = np.uint8
    elif T == UnionFindUInt32:
        dt = np.uint32
    elif T == UnionFindUInt64:
        dt = np.uint64
    s = np.arange(n, dtype=dt)
    for i in range(len(s)):
        s[i] = uf.findLabel(s[i])
    x[:] = s[x]


class TestUnionFind(unittest.TestCase):
    def setUp(self):
        tinyArray = [[10, 10, 0, 0],
                     [0, 10, 0, 10],
                     [10, 10, 0, 10]]
        self.tinyArray = np.asarray(tinyArray).astype(np.uint8)

    def testSimple(self):
        uf = UnionFindArray(np.uint8(3))
        for i in range(4):
            assert uf.findLabel(i) == i
        uf.makeUnion(1, 2)
        assert uf.findLabel(1) == uf.findLabel(2)
        for i in range(3):
            print("{}: {}".format(i, uf.findLabel(i)))
        i = uf.makeContiguous()

    def testNumpyInit(self):
        for dt in (np.uint8, np.uint32, np.uint64):
            print("Trying to initialize UnionFindArray with data type {}".format(dt))
            x = np.arange(4, dtype=dt)
            uf = UnionFindArray(x[-1])
            uf.makeUnion(x[1], x[3])
            assert uf.findLabel(1) == uf.findLabel(3)

    def testLabelAlgorithm(self):
        x = self.tinyArray
        labels = np.zeros(x.shape, dtype=np.uint8)
        uf = UnionFindArray(labels)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] == 0:
                    continue
                currentIndex = uf.nextFreeIndex()
                # left
                if j > 0 and x[i, j-1] == x[i, j]:
                    uf.makeUnion(currentIndex, labels[i, j-1])

                # bottom
                if i > 0 and x[i-1, j] == x[i, j]:
                    uf.makeUnion(currentIndex, labels[i-1, j])

                labels[i, j] = uf.finalizeIndex(currentIndex)

        mapArray(uf, labels)
        print(x)
        print(labels)
        assert labels[-1, -1] != labels[0, 0]
        assert labels[-1, 0] == labels[0, 0]

    def testSafety(self):
        uf = UnionFindArray(np.uint8(127))

    def testNewUF(self):
        a = np.arange(10, dtype=np.uint8)
        uf = UnionFindArray(a[-1])

        def pp(i):
            print("{} -> (Index={}, Label={})".format(
                i, uf.findIndex(i), uf.findLabel(i)))

        uf.makeUnion(a[6], a[7])
        pp(7)
        uf.makeUnion(a[3], a[4])
        pp(7)
        uf.makeContiguous()
        pp(7)
        uf.makeUnion(a[1], a[2])
        pp(7)
        uf.makeContiguous()
        pp(7)

