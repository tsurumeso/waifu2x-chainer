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
            str.append('{}={}'.format(key, self.kwargs[key]))
        return ', '.join(str)

    def append(self, key, value):
        self.kwargs[key] = value
        setattr(self, key, value)


def get_config(base, model, train=True):
    ch = model.ch
    offset = model.offset
    inner_scale = model.inner_scale
    crop_size = base.out_size + offset * 2
    in_size = crop_size // inner_scale

    if train:
        max_size = base.max_size
        patches = base.patches
    else:
        max_size = 0
        coeff = (1 - base.validation_rate) / base.validation_rate
        patches = int(round(base.validation_crop_rate * coeff * base.patches))

    config = {
        'ch': ch,
        'method': base.method,
        'noise_level': base.noise_level,
        'nr_rate': base.nr_rate,
        'chroma_subsampling_rate': base.chroma_subsampling_rate,
        'offset': offset,
        'crop_size': crop_size,
        'in_size': in_size,
        'out_size': base.out_size,
        'inner_scale': inner_scale,
        'max_size': max_size,
        'active_cropping_rate': base.active_cropping_rate,
        'active_cropping_tries': base.active_cropping_tries,
        'random_half_rate': base.random_half_rate,
        'random_color_noise_rate': base.random_color_noise_rate,
        'random_unsharp_mask_rate': base.random_unsharp_mask_rate,
        'patches': patches,
        'downsampling_filters': base.downsampling_filters,
        'resize_blur_min': base.resize_blur_min,
        'resize_blur_max': base.resize_blur_max,
    }
    return Namespace(config)


def set_random_seed(seed, gpu=-1):
    random.seed(seed)
    np.random.seed(seed)
    if gpu >= 0:
        import cupy
        cupy.random.seed(seed)


def load_filelist(dir, shuffle=False):
    files = os.listdir(dir)
    datalist = []
    for file in files:
        path = os.path.join(dir, file)
        if os.path.isfile(path):
            datalist.append(path)
    if shuffle:
        random.shuffle(datalist)
    return datalist
