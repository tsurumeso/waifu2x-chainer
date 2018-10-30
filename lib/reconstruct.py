from __future__ import division

import chainer
import numpy as np
from PIL import Image
import six

from lib import iproc


def _get_padding_size(size, block_size, offset):
    pad = size % block_size
    if pad == 0:
        pad = offset
    else:
        pad = block_size - pad + offset
    return pad


def blockwise(src, model, block_size, batch_size):
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
    xp = model.xp

    inner_block_size = block_size // model.inner_scale
    inner_offset = model.offset // model.inner_scale
    in_block_size = inner_block_size + inner_offset * 2

    in_h, in_w, ch = src.shape
    out_h, out_w = in_h * model.inner_scale, in_w * model.inner_scale
    in_ph = _get_padding_size(in_h, inner_block_size, inner_offset)
    in_pw = _get_padding_size(in_w, inner_block_size, inner_offset)
    out_ph = _get_padding_size(out_h, block_size, model.offset)
    out_pw = _get_padding_size(out_w, block_size, model.offset)

    psrc = np.pad(
        src, ((inner_offset, in_ph), (inner_offset, in_pw), (0, 0)), 'edge')
    nh = (psrc.shape[0] - inner_offset * 2) // inner_block_size
    nw = (psrc.shape[1] - inner_offset * 2) // inner_block_size
    psrc = psrc.transpose(2, 0, 1)

    x = np.zeros((nh * nw, ch, in_block_size, in_block_size), dtype=np.uint8)
    for i in range(0, nh):
        ih = i * inner_block_size
        for j in range(0, nw):
            jw = j * inner_block_size
            psrc_ij = psrc[:, ih:ih + in_block_size, jw:jw + in_block_size]
            x[(i * nw) + j, :, :, :] = psrc_ij

    y = xp.zeros((nh * nw, ch, block_size, block_size), dtype=xp.float32)
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for i in range(0, nh * nw, batch_size):
            batch_x = xp.array(x[i:i + batch_size], dtype=np.float32) / 255
            batch_y = model(batch_x)
            y[i:i + batch_size] = batch_y.data
    y = chainer.backends.cuda.to_cpu(y)

    dst = np.zeros((ch, out_h + out_ph, out_w + out_pw), dtype=np.float32)
    for i in range(0, nh):
        ih = i * block_size
        for j in range(0, nw):
            jw = j * block_size
            dst[:, ih:ih + block_size, jw:jw + block_size] = y[(i * nw) + j]

    dst = dst[:, :out_h, :out_w]
    return dst.transpose(1, 2, 0)


def inv(rot, flip=False):
    if flip:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))[:, ::-1, :]
    else:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))


def get_tta_patterns(src, n):
    src_lr = src.transpose(Image.FLIP_LEFT_RIGHT)
    patterns = [
        [src, None],
        [src.transpose(Image.ROTATE_90), inv(-90)],
        [src.transpose(Image.ROTATE_180), inv(-180)],
        [src.transpose(Image.ROTATE_270), inv(-270)],
        [src_lr, inv(0, True)],
        [src_lr.transpose(Image.ROTATE_90), inv(-90, True)],
        [src_lr.transpose(Image.ROTATE_180), inv(-180, True)],
        [src_lr.transpose(Image.ROTATE_270), inv(-270, True)],
    ]
    if n == 2:
        return [patterns[0], patterns[4]]
    elif n == 4:
        return [patterns[0], patterns[2], patterns[4], patterns[6]]
    elif n == 8:
        return patterns
    return [patterns[0]]


def image_tta(src, model, tta_level, block_size, batch_size):
    inner_scale = model.inner_scale
    dst = np.zeros((src.size[1] * inner_scale, src.size[0] * inner_scale, 3))
    patterns = get_tta_patterns(src, tta_level)
    if model.ch == 1:
        for i, (pat, inv) in enumerate(patterns):
            six.print_(i, end=' ', flush=True)
            pat = np.array(pat.convert('YCbCr'), dtype=np.uint8)
            if i == 0:
                cbcr = pat[:, :, 1:]
            tmp = blockwise(pat[:, :, 0], model, block_size, batch_size)
            if inv is not None:
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
            pat = np.array(pat, dtype=np.uint8)
            tmp = blockwise(pat, model, block_size, batch_size)
            if inv is not None:
                tmp = inv(tmp)
            dst += tmp
        dst /= len(patterns)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
    return dst


def image(src, model, block_size, batch_size):
    if src is None:
        return None
    if model.ch == 1:
        y2rgb = src.mode == 'L'
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(src[:, :, 0], model, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr')
        if y2rgb:
            dst = dst.split()[0]
        else:
            dst = dst.convert('RGB')
    elif model.ch == 3:
        y2rgb = src.mode == 'L'
        if y2rgb:
            src = iproc.y2rgb(src)
        src = np.array(src, dtype=np.uint8)
        dst = blockwise(src, model, block_size, batch_size)
        dst = np.clip(dst, 0, 1) * 255
        dst = Image.fromarray(dst.astype(np.uint8))
        if y2rgb:
            dst = dst.convert('YCbCr').split()[0]
    return dst
