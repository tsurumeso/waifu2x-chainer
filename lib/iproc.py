from __future__ import division
import io

import chainer
import chainer.links as L
import numpy as np
from PIL import Image

try:
    import wand.image
except ImportError:
    pass


def alpha_make_border(rgb, alpha, model):
    xp = model.xp
    sum2d = L.Convolution2D(1, 1, 3, 1, 1, nobias=True, initialW=1)
    if xp == chainer.backends.cuda.cupy:
        sum2d.to_gpu()

    mask = xp.array(alpha, dtype=xp.float32)
    mask[mask > 0] = 1
    mask_nega = xp.abs(mask - 1).astype(xp.uint8) == 1
    eps = 1.0e-7

    rgb = xp.array(rgb, dtype=xp.float32).transpose(2, 0, 1)
    rgb[0][mask_nega] = 0
    rgb[1][mask_nega] = 0
    rgb[2][mask_nega] = 0

    with chainer.no_backprop_mode():
        for _ in range(model.offset):
            mask_weight = sum2d(mask[xp.newaxis, xp.newaxis, :, :]).data[0, 0]
            for i in range(3):
                border = sum2d(rgb[i][xp.newaxis, xp.newaxis, :, :]).data[0, 0]
                border /= (mask_weight + eps)
                rgb[i][mask_nega] = border[mask_nega]
            mask = mask_weight
            mask[mask > 0] = 1
            mask_nega = xp.abs(mask - 1).astype(xp.uint8) == 1
    rgb = chainer.backends.cuda.to_cpu(xp.clip(rgb, 0, 255))
    return Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))


def y2rgb(src):
    rgb = np.zeros((src.size[1], src.size[0], 3))
    rgb[:, :, 0] = np.array(src)
    rgb[:, :, 1] = np.array(src)
    rgb[:, :, 2] = np.array(src)
    return Image.fromarray(rgb.astype(np.uint8))


def read_image_rgb_uint8(path):
    src = Image.open(path)
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    mode = src.mode
    if mode in ('LA', 'RGBA'):
        if mode == 'LA':
            src = src.convert('RGBA')
        rgb = Image.new('RGB', src.size, (128, 128, 128))
        rgb.paste(src, mask=src.split()[-1])
    else:
        rgb = src.convert('RGB')
    dst = np.array(rgb, dtype=np.uint8)
    return dst


def array_to_wand(src):
    assert isinstance(src, np.ndarray)
    with io.BytesIO() as buf:
        tmp = Image.fromarray(src).convert('RGB')
        tmp.save(buf, 'PNG', compress_level=0)
        dst = wand.image.Image(blob=buf.getvalue())
    return dst


def wand_to_array(src):
    assert isinstance(src, wand.image.Image)
    with io.BytesIO(src.make_blob('PNG')) as buf:
        tmp = Image.open(buf).convert('RGB')
        dst = np.array(tmp, dtype=np.uint8)
    return dst


def nn_scaling(src, ratio):
    if src is None:
        return None

    if isinstance(src, Image.Image):
        w, h = src.size[:2]
        dst = src.resize((int(w * ratio), int(h * ratio)), Image.NEAREST)
    else:
        with array_to_wand(src) as tmp:
            h, w = src.shape[:2]
            tmp.resize(int(w * ratio), int(h * ratio), 'box')
            dst = wand_to_array(tmp)
    return dst


def jpeg(src, sampling_factor='1x1,1x1,1x1', quality=90):
    src.format = 'jpg'
    src.compression_quality = quality
    src.options['jpeg:sampling-factor'] = sampling_factor
    return wand.image.Image(blob=src.make_blob())


def pcacov(x):
    imcol = x.reshape(3, x.shape[0] * x.shape[1])
    ce, cv = np.linalg.eigh(np.cov(imcol))
    return ce, cv


def clipped_psnr(y, t, a_min=0., a_max=1.):
    xp = chainer.backends.cuda.get_array_module(y)
    y_c = xp.clip(y, a_min, a_max)
    t_c = xp.clip(t, a_min, a_max)
    mse = xp.mean(xp.square(y_c - t_c))
    psnr = 20 * xp.log10(a_max / xp.sqrt(mse))
    return psnr
