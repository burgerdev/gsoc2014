#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import unittest
import numpy as np

from lazycc import mergeLabels, UnionFindArray


class TestMergeLabels(unittest.TestCase):
    def setUp(self):
        tinyArray = [[1, 0, 0],
                     [1, 0, 2],
                     [0, 0, 2]]
        self.tinyArray = np.asarray(tinyArray).astype(np.uint8)

    def testSimple(self):
        plane = np.asarray(self.tinyArray)
        planeInc = plane * 3
        uf = UnionFindArray(plane)
        ufInc = UnionFindArray(planeInc)
        guf = UnionFindArray()

        mergeLabels(plane, planeInc, uf, ufInc, guf)
        print(uf)
        print(ufInc)
        print(guf)
        uf.mapArray(plane)
        ufInc.mapArray(planeInc)
        print(plane)
        print(planeInc)

        np.testing.assert_array_equal(plane, planeInc)
        assert plane[0, 0] > 0
        assert plane[-1, -1] > 0


class TestUnionFind(unittest.TestCase):
    def setUp(self):
        tinyArray = [[10, 10, 0, 0],
                     [0, 10, 0, 10],
                     [10, 10, 0, 10]]
        self.tinyArray = np.asarray(tinyArray).astype(np.uint8)

    def testSimple(self):
        x = self.tinyArray
        labels = np.zeros(x.shape, dtype=np.uint32)
        uf = UnionFindArray()

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                print(labels)
                print("Processing {}".format((i, j)))
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

        print(labels)
        uf.mapArray(labels)
        print(labels)
        print(uf)
        assert labels[-1, -1] != labels[0, 0]
        assert labels[-1, 0] == labels[0, 0]
