#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np

## mockup for vigra's UnionFindArray structure
class UnionFindArray(object):

    def __init__(self, array=None):
        if array is not None:
            values = sorted(np.unique(array))
            self._map = dict(zip(values, values))
            self._map[0] = 0
        else:
            self._map = {0: 0}

    def makeUnion(self, a, b):
        if a < b:
            self._map[b] = a
            if a not in self._map:
                self._map[a] = a
        else:
            self._map[a] = b
            if b not in self._map:
                self._map[b] = b

    def finalizeLabel(self, a):
        raise NotImplementedError()

    def makeContiguous(self):
        raise NotImplementedError()

    def nextFreeLabel(self):
        raise NotImplementedError()

    def makeNewLabel(self):
        newLabel = max(self._map.keys())+1
        self._map[newLabel] = newLabel
        return newLabel

    def find(self, a):
        _a = self._map[a]
        if a != _a:
            return self.find(_a)
        else:
            return a

    def mapArray(self, arr):
        x = np.zeros((max(self._map.keys())+1,))
        for key in self._map:
            x[key] = self._map[key]
        x = np.abs(x).astype(np.uint32)
        arr[:] = x[arr]

    def __str__(self):
        s = "<UnionFindArray>\n{}".format(self._map)
        return s
