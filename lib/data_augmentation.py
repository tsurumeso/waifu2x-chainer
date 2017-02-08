from __future__ import division

import random
import numpy as np
from PIL import Image
from PIL import ImageFilter

from lib import iproc


def random_unsharp_mask(src, p):
    if np.random.uniform() < p:
        tmp = Image.fromarray(src)
        radius = random.randint(1, 3)
        percent = random.randint(10, 90)
        threshold = random.randint(0, 5)
        unsharp = ImageFilter.UnsharpMask(radius=radius,
                                          percent=percent,
                                          threshold=threshold)
        dst = np.array(tmp.filter(unsharp), dtype=np.uint8)
        return dst
    else:
        return src


def random_flip(src):
    rand = random.randint(0, 3)
    dst = src
    if rand == 0:
        dst = src[::-1, :, :]
    elif rand == 1:
        dst = src[:, ::-1, :]
    elif rand == 2:
        dst = src[::-1, ::-1, :]
    return dst


def random_half(src, p):
    # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
    # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
    # 'sinc'
    if np.random.uniform() < p:
        with iproc.array_to_wand(src) as tmp:
            filter = ('box', 'box', 'blackman', 'cubic', 'lanczos')
            h, w = src.shape[:2]
            rand = random.randint(0, len(filter) - 1)
            tmp.resize(w // 2, h // 2, filter[rand])
            dst = iproc.wand_to_array(tmp)
        return dst
    else:
        return src


def random_shift_1px(src):
    direction = random.randint(0, 3)
    x_shift = 0
    y_shift = 0
    if direction == 0:
        x_shift = 1
        y_shift = 0
    elif direction == 1:
        x_shift = 0
        y_shift = 1
    elif direction == 2:
        x_shift = 1
        y_shift = 1
    w = src.shape[1] - x_shift
    h = src.shape[0] - y_shift
    w = w - (w % 4)
    h = h - (h % 4)
    dst = src[y_shift:y_shift + h, x_shift:x_shift + w, :]
    return dst
