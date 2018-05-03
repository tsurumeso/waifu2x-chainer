from __future__ import division
import io

import chainer
from chainer import cuda
import chainer.links as L
import numpy as np
from PIL import Image

try:
    from wand.image import Image as WandImage
except(ModuleNotFoundError, ImportError):
    pass


def alpha_make_border(rgb, alpha, offset):
    sum2d = L.Convolution2D(1, 1, 3, 1, 1, nobias=True)
    sum2d.W.data = np.ones((1, 1, 3, 3))

    mask = np.array(alpha, dtype=np.float32)
    mask[mask > 0] = 1
    mask_nega = np.abs(mask - 1).astype(np.uint8) == 1
    eps = 1.0e-7

    rgb = np.array(rgb, dtype=np.float32).transpose(2, 0, 1)
    rgb[0][mask_nega] = 0
    rgb[1][mask_nega] = 0
    rgb[2][mask_nega] = 0

    with chainer.no_backprop_mode():
        for _ in range(offset):
            mask_weight = sum2d(mask[np.newaxis, np.newaxis, :, :]).data[0, 0]
            for i in range(3):
                border = sum2d(rgb[i][np.newaxis, np.newaxis, :, :]).data[0, 0]
                border /= (mask_weight + eps)
                rgb[i][mask_nega] = border[mask_nega]
            mask = mask_weight
            mask[mask > 0] = 1
            mask_nega = np.abs(mask - 1).astype(np.uint8) == 1
    rgb = np.clip(rgb, 0, 255)
    return Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))


def y2rgb(src):
    rgb = np.zeros((src.size[1], src.size[0], 3))
    rgb[:, :, 0] = np.array(src)
    rgb[:, :, 1] = np.array(src)
    rgb[:, :, 2] = np.array(src)
    return Image.fromarray(rgb.astype(np.uint8))


def read_image_rgb_uint8(path):
    src = Image.open(path).convert('RGB')
    dst = np.array(src, dtype=np.uint8)
    return dst


def array_to_wand(src):
    assert isinstance(src, np.ndarray)
    with io.BytesIO() as buf:
        tmp = Image.fromarray(src).convert('RGB')
        tmp.save(buf, 'PNG', compress_level=0)
        dst = WandImage(blob=buf.getvalue())
    return dst


def wand_to_array(src):
    assert isinstance(src, WandImage)
    with io.BytesIO(src.make_blob('PNG')) as buf:
        tmp = Image.open(buf).convert('RGB')
        dst = np.array(tmp, dtype=np.uint8)
    return dst


def scale(src, ratio, filter='box'):
    with array_to_wand(src) as tmp:
        h, w = src.shape[:2]
        tmp.resize(int(w * ratio), int(h * ratio), filter)
        dst = wand_to_array(tmp)
    return dst


def scale2x(src):
    if src is None:
        return None
    dst = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
    return dst


def jpeg(src, sampling_factor='1x1,1x1,1x1', quality=90):
    src.format = 'jpg'
    src.compression_quality = quality
    src.options['jpeg:sampling-factor'] = sampling_factor
    return WandImage(blob=src.make_blob())


def pcacov(x):
    imcol = x.reshape(3, x.shape[0] * x.shape[1])
    ce, cv = np.linalg.eigh(np.cov(imcol))
    return ce, cv


def clipped_psnr(y, t, a_min=0., a_max=1.):
    xp = cuda.get_array_module(y)
    y_c = xp.clip(y, a_min, a_max)
    t_c = xp.clip(t, a_min, a_max)
    mse = xp.mean(xp.square(y_c - t_c))
    psnr = 20 * xp.log10(a_max / xp.sqrt(mse))
    return psnr
