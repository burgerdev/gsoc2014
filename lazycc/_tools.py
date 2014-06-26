#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

from collections import defaultdict
import numpy as np


class InfiniteLabelIterator(object):

    def __init__(self, n, dtype=np.uint32):
        if not np.issubdtype(dtype, np.integer):
            raise ValueError("Labels must have an integral type")
        self.dtype = dtype
        self.n = n

    def next(self):
        a = self.dtype(self.n)
        assert a < np.iinfo(self.dtype).max, "Label overflow."
        self.n += 1
        return a


class LabelGraph(object):

    def __init__(self, shape):
        self._d = len(shape)
        self._shape = shape
        self._map = defaultdict()

    def getEdges(self, v, w):
        vert, d = index2dim(v, w)

        x = self._map
        if x is None:
            return None

        edges = x[d]
        if edges is None:
            return None

        # invert edges if vertices are in the wrong order
        if vert == w:
            edges = list(map(lambda edge: tuple(reversed(edge)), edges))

        return edges

    def appendEdges(self, v, w, edges):
        vert, d = index2dim(v, w)

        if self._map[vert] is None:
            self._map[vert] = np.empty((self._d,), dtype=np.object)

        # invert edges if vertices are in the wrong order
        if vert == w:
            edges = set(map(lambda edge: tuple(reversed(edge)), edges))

        if self._map[vert][d] is None:
            self._map[vert][d] = set()
        self._map[vert][d].union(edges)


def dim2Index(v, d):
    v = np.asarray(v, dtype=np.int)
    v[d] += 1


def index2dim(v, w):
    for i in range(len(v)):
        if v[i] - w[i] == 1:
            return (w, i)
        elif v[i] - w[i] == -1:
            return (v, i)
    return (None, None)
