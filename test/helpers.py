#!/usr/bin/env python
# coding: utf-8
# author: Markus Döring

import numpy as np
from lazyflow.operator import Operator, InputSlot, OutputSlot


def assertEquivalentLabeling(labelImage, referenceImage):
    x = labelImage
    y = referenceImage
    assert np.all(x.shape == y.shape),\
        "Shapes do not agree ({} vs {})".format(x.shape, y.shape)

    # identify labels used in x
    labels = set(x.flat)
    for label in labels:
        if label == 0:
            continue
        idx = np.where(x == label)
        refblock = y[idx]
        # check that labels are the same
        corner = [a[0] for a in idx]
        print("Inspecting region of size {} at {}".format(refblock.size, corner))

        assert np.all(refblock == refblock[0]),\
            "Uniformly labeled region at coordinates {} has more than one label in the reference image".format(corner)
        # check that nothing else is labeled with this label
        m = refblock.size
        n = (y == refblock[0]).sum()
        assert m == n, "There are more pixels with (reference-)label {} than pixels with label {}.".format(refblock[0], label)

    assert len(labels) == len(set(y.flat)), "The number of labels does not agree, perhaps some region was missed"


class DirtyAssert(Operator):
    Input = InputSlot()

    def willBeDirty(self, t, c):
        self._t = t
        self._c = c

    def propagateDirty(self, slot, subindex, roi):
        t_ind = self.Input.meta.axistags.index('t')
        c_ind = self.Input.meta.axistags.index('c')
        assert roi.start[t_ind] == self._t
        assert roi.start[c_ind] == self._c
        assert roi.stop[t_ind] == self._t+1
        assert roi.stop[c_ind] == self._c+1
        raise self.PropagateDirtyCalled()

    class PropagateDirtyCalled(Exception):
        pass
