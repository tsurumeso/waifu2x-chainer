from __future__ import division

import six
import numpy as np
from PIL import Image
from chainer import cuda

from lib import iproc
from lib import utils


def blockwise(model, src, block_size, batch_size):
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


def get_tta_patterns(src):
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
    return patterns 


def scale_tta(model, src, block_size, batch_size):
    patterns = get_tta_patterns(src)
    dst = np.zeros((src.size[1] * 2, src.size[0] * 2, 3))
    for i, (src, inv) in enumerate(patterns):
        six.print_(i, end=' ', flush=True)
        src = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        tmp = blockwise(model, src, block_size, batch_size)
        if not inv is None:
            tmp = inv(tmp)
        dst += tmp
    dst /= len(patterns)
    dst = np.clip(dst, 0, 1) * 255
    dst = Image.fromarray(dst.astype(np.uint8))
    return dst


def noise_tta(model, src, block_size, batch_size):
    patterns = get_tta_patterns(src)
    dst = np.zeros((src.size[1], src.size[0], 3))
    for i, (src, inv) in enumerate(patterns):
        six.print_(i, end=' ', flush=True)
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        tmp = blockwise(model, src, block_size, batch_size)
        if not inv is None:
            tmp = inv(tmp)
        dst += tmp
    dst /= len(patterns)
    dst = np.clip(dst, 0, 1) * 255
    dst = Image.fromarray(dst.astype(np.uint8))
    return dst


def scale(model, src, block_size, batch_size):
    src = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
    if model.ch == 1:
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(model, src[:, :, 0], block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr').convert('RGB')
        return dst
    elif model.ch == 3:
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        dst = blockwise(model, src, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
        return dst


def noise(model, src, block_size, batch_size):
    if model.ch == 1:
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(model, src[:, :, 0], block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr').convert('RGB')
        return dst
    elif model.ch == 3:
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        dst = blockwise(model, src, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
        return dst
