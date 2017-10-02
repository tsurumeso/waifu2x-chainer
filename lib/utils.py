from __future__ import print_function
import os
import random

import chainer
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


def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__:
            continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child):
            continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)
