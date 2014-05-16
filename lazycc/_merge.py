#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np

## join the labels of two adjacent chunks
#
# The boundary of the two chunks A and B is inspected and the UnionFind
# structures are changed such that connected components have the same global 
# label. The labels must already be global, i.e. UF_a/UF_b point
# the local labels in A/B to valid labels in GUF.
#
# @param hyperplane_a boundary in A (const)
# @param hyperplane_b boundary in B (const)
# @param label_hyperplane_a label boundary in A (const)
# @param label_hyperplane_b label boundary in B (const)
# @param UF_a hash table, mapping local labels to global labels (const)
# @param UF_b hash table, mapping local labels to global labels (const)
# @param GUF global UnionFind
# @returns None
def mergeLabels(hyperplane_a, hyperplane_b,
                label_hyperplane_a, label_hyperplane_b,
                mapping_a, mapping_b, GUF):
    
    # the indices where objects are adjacent
    idx = hyperplane_a == hyperplane_b

    # merge each pair of labels
    for label_a, label_b in zip(label_hyperplane_a[idx],
                                label_hyperplane_b[idx]):
        GUF.makeUnion(mapping_a[label_a], mapping_b[label_b])
