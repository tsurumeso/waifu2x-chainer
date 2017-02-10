from __future__ import division

import six
import numpy as np
from PIL import Image
from chainer import cuda

from lib import iproc
from lib import utils


def blockwise(src, model, block_size, batch_size):
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
    h, w, ch = src.shape
    offset = utils.offset_size(model)
    xp = utils.get_model_module(model)
    src = src.transpose(2, 0, 1) / 255.

    ph = block_size - (h % block_size) + offset // 2
    pw = block_size - (w % block_size) + offset // 2
    src = np.array([np.pad(x, ((offset // 2, ph), (offset // 2, pw)), 'edge')
                    for x in src])
    nh = src.shape[1] // block_size
    nw = src.shape[2] // block_size

    block_offset = block_size + offset
    x = np.ndarray((nh * nw, ch, block_offset, block_offset), dtype=np.float32)
    for i in range(0, nh):
        ih = i * block_size
        for j in range(0, nw):
            index = (i * nw) + j
            jw = j * block_size
            src_ij = src[:, ih:ih + block_offset, jw:jw + block_offset]
            x[index, :, :, :] = src_ij

    y = np.ndarray((nh * nw, ch, block_size, block_size), dtype=np.float32)
    for i in range(0, nh * nw, batch_size):
        x_batch = xp.array(x[i:i + batch_size])
        output = model(x_batch)
        y[i:i + batch_size] = cuda.to_cpu(output.data)

    dst = np.ndarray((ch, h + ph, w + pw), dtype=np.float32)
    for i in range(0, nh):
        ih = i * block_size
        for j in range(0, nw):
            index = (i * nw) + j
            jw = j * block_size
            dst[:, ih:ih + block_size, jw:jw + block_size] = y[index]

    dst = dst[:, :h, :w]
    return dst.transpose(1, 2, 0)


def get_tta_patterns(src, n):
    src_lr = src.transpose(Image.FLIP_LEFT_RIGHT)
    patterns = [[src, None],
               [src.transpose(Image.ROTATE_90), iproc.inv(-90)],
               [src.transpose(Image.ROTATE_180), iproc.inv(-180)],
               [src.transpose(Image.ROTATE_270), iproc.inv(-270)],
               [src_lr, iproc.inv(0, True)],
               [src_lr.transpose(Image.ROTATE_90), iproc.inv(-90, True)],
               [src_lr.transpose(Image.ROTATE_180), iproc.inv(-180, True)],
               [src_lr.transpose(Image.ROTATE_270), iproc.inv(-270, True)],
    ]
    if n == 2:
        return [patterns[0], patterns[4]]
    elif n == 4:
        return [patterns[0], patterns[2], patterns[4], patterns[6]]
    elif n == 8:
        return patterns
    return [patterns[0]]


def image_tta(src, model, scale, tta_level, block_size, batch_size):
    if scale:
        src = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
    patterns = get_tta_patterns(src, tta_level)
    dst = np.zeros((src.size[1], src.size[0], 3))
    cbcr = np.zeros((src.size[1], src.size[0], 2))
    if model.ch == 1:
        for i, (pat, inv) in enumerate(patterns):
            six.print_(i, end=' ', flush=True)
            pat = np.array(pat.convert('YCbCr'), dtype=np.uint8)
            if i == 0:
                cbcr = pat[:, :, 1:]
            tmp = blockwise(pat[:, :, 0], model, block_size, batch_size)
            if not inv is None:
                tmp = inv(tmp)
            dst[:, :, 0] += tmp[:, :, 0]
        dst /= len(patterns)
        dst = np.clip(dst, 0, 1) * 255
        dst[:, :, 1:] = cbcr
        dst = dst.astype(np.uint8)
        dst = Image.fromarray(dst, mode='YCbCr').convert('RGB')
    elif model.ch == 3:
        for i, (pat, inv) in enumerate(patterns):
            six.print_(i, end=' ', flush=True)
            pat = np.array(pat.convert('RGB'), dtype=np.uint8)
            tmp = blockwise(pat, model, block_size, batch_size)
            if not inv is None:
                tmp = inv(tmp)
            dst += tmp
        dst /= len(patterns)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
    return dst


def image(src, model, scale, block_size, batch_size):
    if scale:
        src = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
    if model.ch == 1:
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(src[:, :, 0], model, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr').convert('RGB')
    elif model.ch == 3:
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        dst = blockwise(src, model, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
    return dst
