#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
from collections import defaultdict
from functools import partial, wraps

from threading import Lock

from _tools import InfiniteLabelIterator

def locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


## mockup for vigra's UnionFindArray structure
class UnionFindArray(object):

    def __init__(self, nextFree=1):
        self._map = dict(zip(*(xrange(nextFree),)*2))
        self._lock = Lock()
        self._nextFree = nextFree
        self._it = None

    ## join regions a and b
    # callback is called whenever two regions are joined that were
    # separate before, signature is 
    #   callback(smaller_label, larger_label)
    @locked
    def makeUnion(self, a, b):
        assert a in self._map
        assert b in self._map

        a = self._findIndex(a)
        b = self._findIndex(b)

        # avoid cycles by choosing the smallest label as the common one
        # swap such that a is smaller
        if a > b:
            a, b = b, a

        self._map[b] = a

    def finalizeLabel(self, a):
        raise NotImplementedError()

    def makeContiguous(self):
        raise NotImplementedError()

    def nextFreeLabel(self):
        raise NotImplementedError()

    @locked
    def makeNewIndex(self):
        newLabel = self._nextFree
        self._nextFree += 1
        self._map[newLabel] = newLabel
        return newLabel

    @locked
    def findIndex(self, a):
        return self._findIndex(a)

    def _findIndex(self, a):
        while a != self._map[a]:
            a = self._map[a]
        return a

    def __str__(self):
        s = "<UnionFindArray>\n{}".format(self._map)
        return s

    def __getitem__(self, key):
        raise NotImplementedError()
