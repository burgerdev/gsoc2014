#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np

#from _mockup import UnionFindArray
from _lazycc_cxx import UnionFindUInt8, UnionFindUInt32, UnionFindUInt64

def UnionFindArray(x):
    if isinstance(x, np.ndarray):
        a = max(x.flat) + 1
    else:
        a = x
    if x.dtype == np.uint8:
        return UnionFindUInt8(np.uint8(a))
    elif x.dtype == np.uint32:
        return UnionFindUInt32(np.uint32(a))
    elif x.dtype == np.uint64:
        return UnionFindUInt64(np.uint64(a))
    else:
        raise ValueError("Unsupported dtype {} for UnionFindArray".format(x.dtype))


def _dtypeFromType(T):
    if T == UnionFindUInt8:
        return np.uint8
    elif T == UnionFindUInt32:
        return np.uint32
    elif T == UnionFindUInt64:
        return np.uint64
    else:
        raise ValueError()


from _lazycc_cxx import mergeLabels
#from _merge import mergeLabels
from _opLazyCC import OpLazyCC
from _opBlockwiseCC import OpBlockwiseCC
from _tools import LabelGraph, index2dim, dim2Index
