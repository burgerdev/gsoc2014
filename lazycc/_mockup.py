#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
from collections import defaultdict

## mockup for vigra's UnionFindArray structure
class UnionFindArray(object):

    def __init__(self, array=None):
        if array is not None:
            values = sorted(np.unique(array))
            self._map = dict(zip(values, values))
            self._map[0] = 0
        else:
            self._map = {0: 0}

        self._offset = 0

    ## join regions a and b
    # callback is called whenever two regions are joined that were
    # separate before, signature is 
    #   callback(smaller_label, larger_label)
    def makeUnion(self, a, b, callback=None):
        assert a in self._map
        assert b in self._map
        
        a = self.find(a, useOffset=False)
        b = self.find(b, useOffset=False)
        
        # avoid cycles by choosing the smallest label as the common one
        # swap such that a is smaller
        if a > b:
            a, b = b, a

        self._map[b] = a
        
        if a != b and callback is not None:
            callback(a, b)
        
        
        

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

    def find(self, a, useOffset=True):
        if a == 0:
            return 0
        _a = self._map[a]
        if a != _a:
            return self.find(_a)
        else:
            return _a + (self._offset if useOffset else 0)

    def mapArray(self, arr):
        x = np.zeros((max(self._map.keys())+1,))
        for key in self._map:
            x[key] = self.find(key)
        x = np.abs(x).astype(np.uint32)
        arr[:] = x[arr]

    def __str__(self):
        s = "<UnionFindArray>\n{}".format(self._map)
        return s

    def setOffset(self, n):
        self._offset = n

    def __getitem__(self, key):
        return self.find(key)
