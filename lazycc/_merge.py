#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring


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
                UF_a, UF_b, GUF):

    # iterate over all pixels
    for x, y, a, b in zip(hyperplane_a.flat, hyperplane_b.flat,
                          label_hyperplane_a.flat, label_hyperplane_b.flat):

        if a == 0 or b == 0 or x != y:
            # no real traversal
            continue

        if _isGlobal(a, UF_a):
            if _isGlobal(b, UF_b):
                print("Merging local labels {} and {}".format(a, b))
                GUF.makeUnion(UF_a[a],
                              UF_b[b])
            else:
                # assign A's global label to B
                UF_b.setGlobal(b, UF_a.find(a))

        else:
            if _isGlobal(b, UF_b):
                # assign B's global label to A
                UF_a.setGlobal(a, UF_b.find(b))
            else:
                # assign a new global label to both
                label = _getGlobalLabel(GUF)
                UF_a.setGlobal(a, label)
                UF_b.setGlobal(b, label)


def _isGlobal(x, uf):
    #return uf.isGlobal(x)
    return True


def _getGlobalLabel(uf):
    return uf.makeNewLabel()

