from __future__ import print_function

import os
import six
import random
import chainer


class Namespace():

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


def get_config(args, ch, offset, train=True):
    if train:
        max_size = args.max_size
        patches = args.patches
    else:
        max_size = 0
        coeff = (1 - args.validation_rate) / args.validation_rate
        patches = int(round(args.validation_crop_rate * coeff * args.patches))

    config = {
        'ch': ch,
        'method': args.method,
        'noise_level': args.noise_level,
        'nr_rate': args.nr_rate,
        'chroma_subsampling_rate': args.chroma_subsampling_rate,
        'offset': offset,
        'insize': args.crop_size + offset,
        'crop_size': args.crop_size,
        'max_size': max_size,
        'active_cropping_rate': args.active_cropping_rate,
        'active_cropping_tries': args.active_cropping_tries,
        'random_half_rate': args.random_half_rate,
        'random_color_noise_rate': args.random_color_noise_rate,
        'random_unsharp_mask_rate': args.random_unsharp_mask_rate,
        'patches': patches,
        'downsampling_filters': args.downsampling_filters,
        'resize_blur_min': args.resize_blur_min,
        'resize_blur_max': args.resize_blur_max,
    }
    return Namespace(config)


def get_model_module(model):
    if isinstance(model, chainer.Chain):
        child = six.next(model.children())
        return child.xp


def load_datalist(dir, shuffle=False):
    files = os.listdir(dir)
    datalist = []
    for file in files:
        datalist.append(os.path.join(dir, file))
    if shuffle:
        random.shuffle(datalist)
    return datalist


def offset_size(model):
    offset = 0
    if hasattr(model, 'offset'):
        offset = model.offset
    else:
        for child in model.children():
            if isinstance(child, chainer.Link):
                offset += child.W.data.shape[2] - 1
    return offset


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
