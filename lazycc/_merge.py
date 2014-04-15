#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring


## merges the UnionFind structures of two adjacent chunks
#
# The boundary of the two chunks A and B is inspected and the UnionFind
# structures are changed such that connected components have the same global 
# label. All labels that remain local after the procedure do not traverse the
# chunk boundary.
#
# @param hyperplane_a boundary in A
# @param hyperplane_b boundary in B
# @param UF_a local UnionFind of A
# @param UF_a local UnionFind of B
# @param GUF global UnionFind
# @returns None
def mergeLabels(hyperplane_a, hyperplane_b, UF_a, UF_b, GUF):

    # iterate over all pixels
    for a, b in zip(hyperplane_a.flat, hyperplane_b.flat):

        if a == 0 or b == 0:
            # no real traversal
            continue

        if _isGlobal(a, UF_a):
            if _isGlobal(b, UF_b):
                print("Merging {} and {}".format(UF_a.find(a), UF_b.find(b)))
                GUF.makeUnion(UF_a.find(a),
                              UF_b.find(b))
            else:
                # assign A's global label to B
                UF_b.makeUnion(b, UF_a.find(a))

        else:
            if _isGlobal(b, UF_b):
                # assign B's global label to A
                UF_a.makeUnion(a, UF_b.find(b))
            else:
                # assign a new global label to both
                label = _getGlobalLabel(GUF)
                UF_a.makeUnion(a, label)
                UF_b.makeUnion(b, label)


def _isGlobal(x, uf):
    return uf.find(x) < 0


def _getGlobalLabel(uf):
    return -uf.makeNewLabel()
