#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import unittest

from lazycc import LabelGraph


class TestLabelGraph(unittest.TestCase):

    def setUp(self):
        pass

    def testIndexToDim(self):
        from lazycc import index2dim
        v = (2, 1, 3)
        w = (2, 2, 3)
        vert, d = index2dim(v, w)
        print(vert)
        assert vert is v
        assert d == 1

        vert, d = index2dim(w, v)
        print(vert)
        assert vert is v
        assert d == 1