from __future__ import print_function
import os
import random

import numpy as np


class Namespace(object):

    def __init__(self, kwargs):
        self.kwargs = kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def __repr__(self):
        str = []
        for key in self.kwargs.keys():
            str.append('%s=%s' % (key, self.kwargs[key]))
        return ', '.join(str)

    def append(self, key, value):
        self.kwargs[key] = value
        setattr(self, key, value)


def set_random_seed(seed, gpu=-1):
    random.seed(seed)
    np.random.seed(seed)
    if gpu >= 0:
        import cupy
        cupy.random.seed(seed)


def load_datalist(dir, shuffle=False):
    files = os.listdir(dir)
    datalist = []
    for file in files:
        datalist.append(os.path.join(dir, file))
    if shuffle:
        random.shuffle(datalist)
    return datalist
