from typing import Union, Iterable

import numpy as np
import h5py
import os
import scipy as sp
from matplotlib import pyplot as plt
from numpy import ndarray


class Kulite(object):
    data: ndarray

    def __init__(self, path, h5file = 'Test.h5', h5grp = '2022_02_01/msg_001/'):

        self.path = path

        self.h5file = self.path + h5file
        self.h5grp = h5grp
        self.h5 = h5py.File(self.h5file, 'r+')

        return

    def load(self, name, start=0, end=None):

        try:
            self.data = np.load(self.path + name)[start:end]

        finally:
            return print('%s could not be found. Check Path' % name)
            # integrate h5 reading

        return

    def load_h5(self):
        self.data = self.h5[self.h5grp + '/Kulites']
        return

