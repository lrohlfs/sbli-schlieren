from typing import Union, Iterable

import numpy as np
import os
import scipy as sp
from matplotlib import pyplot as plt
from numpy import ndarray


class Kulite(object):
    data: ndarray

    def __init__(self, path):

        self.path = path

        return

    def load(self, name, start=0, end=None):

        try:
            self.data = np.load(self.path + name)[start:end]

        finally:
            return print('%s could not be found. Check Path' % name)
            # integrate h5 reading

        return
