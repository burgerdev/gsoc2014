#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import unittest
import numpy as np
import vigra

from lazycc import mergeLabels, UnionFindArray
from helpers import assertEquivalentLabeling


class TestMergeLabels(unittest.TestCase):
    def setUp(self):
        tinyArray = [[1, 0, 0],
                     [1, 0, 2],
                     [0, 0, 2]]
        self.tinyArray = np.asarray(tinyArray).astype(np.uint8)

        uArray = np.zeros((10, 10), dtype=np.uint8)
        uLabels = np.zeros(uArray.shape, dtype=np.uint32)

        # horse shoe
        uArray[[2, 8], 4:9] = 255
        uLabels[[2, 8], 4:9] = 1
        uArray[2:9, 8] = 255
        uLabels[2:9:, 8] = 1

        # square in top left corner
        uArray[0:3, 0:3] = 255
        uLabels[0:3, 0:3] = 2

        # line crossing a border
        uArray[4:6, 1] = 255
        uLabels[4:6, 1] = 3

        # square in the middle
        uArray[4:6, 4:6] = 255
        uLabels[4:6, 4:6] = 4

        # line ending at boundary
        uArray[0, 4] = 13
        uLabels[0, 4] = 5
        uArray[0, 5] = 31
        uLabels[0, 5] = 6

        self.uLabels = uLabels
        self.uArray = uArray

    def testSimple(self):
        plane = np.asarray(self.tinyArray)
        labels = plane.copy()
        planeInc = plane.copy()
        labelsInc = labels.copy()
        labelsInc[planeInc > 0] += 1

        uf = UnionFindArray(labels)
        ufInc = UnionFindArray(labelsInc)
        ufInc.setOffset(2)
        guf = UnionFindArray()
        for i in range(5):
            guf.makeNewLabel()

        mergeLabels(plane, planeInc, labels, labelsInc, uf, ufInc, guf)
        uf.mapArray(labels)
        ufInc.mapArray(labelsInc)
        guf.mapArray(labels)
        guf.mapArray(labelsInc)

        np.testing.assert_array_equal(labels, labelsInc)
        plane *= planeInc
        assert plane[0, 0] > 0
        assert plane[-1, -1] > 0

    def testBlockwiseCC(self):
        x = self.uArray
        m = x.shape[0]//2
        labels = self.uLabels
        res = np.zeros(x.shape, dtype=np.uint32)
        uf = np.empty((2, 2), dtype=np.object)
        guf = UnionFindArray()

        # label each block independently
        s = 0
        for i in range(2):
            for j in range(2):
                res[i*m:(i+1)*m, j*m:(j+1)*m] = vigra.analysis.labelImageWithBackground(x[i*m:(i+1)*m, j*m:(j+1)*m])
                uf[i, j] = UnionFindArray(res[i*m:(i+1)*m, j*m:(j+1)*m])
                offset = np.max(res[i*m:(i+1)*m, j*m:(j+1)*m])
                for k in range(offset):
                    guf.makeNewLabel()
                uf[i, j].setOffset(s)
                s += offset

        # merge blocks
        mergeLabels(x[:m, m-1], x[:m, m], res[:m, m-1], res[:m, m],
                    uf[0, 0], uf[0, 1], guf)
        mergeLabels(x[m-1, :m], x[m, :m], res[m-1, :m], res[m, :m],
                    uf[0, 0], uf[1, 0], guf)
        mergeLabels(x[m:, m-1], x[m:, m], res[m:, m-1], res[m:, m],
                    uf[1, 0], uf[1, 1], guf)
        mergeLabels(x[m-1, m:], x[m, m:], res[m-1, m:], res[m, m:],
                    uf[0, 1], uf[1, 1], guf)
        print("Result before any mapping")
        print(res)
        for i in range(2):
            for j in range(2):
                uf[i, j].mapArray(res[i*m:(i+1)*m, j*m:(j+1)*m])
        print("Result after local mapping (and GUF)")
        print(res)
        print(guf)
        guf.mapArray(res)
        print("Result after global mapping")

        print(res)

        print("Original volume and labels")
        print(labels)
        print(x)

        assertEquivalentLabeling(res, labels)


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

        uf.mapArray(labels)
        assert labels[-1, -1] != labels[0, 0]
        assert labels[-1, 0] == labels[0, 0]

