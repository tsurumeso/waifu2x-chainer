from __future__ import division

import io
import numpy as np
from chainer import cuda
from PIL import Image
from wand.image import Image as WandImage


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


def jpeg(src, sampling_factor='1x1,1x1,1x1', quality=90):
    src.format = 'jpg'
    src.compression_quality = quality
    src.options['jpeg:sampling-factor'] = sampling_factor
    return WandImage(blob=src.make_blob())


def inv(rot, flip=False):
    if flip:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))[:, ::-1, :]
    else:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))


def to_image(data, ch, batch=False):
    img = cuda.to_cpu(data)
    if batch:
        img = np.clip(img, 0, 1) * 255
    if ch == 1:
        return Image.fromarray(img[0].astype(np.uint8))
    elif ch == 3:
        img = img.transpose(1, 2, 0)
        return Image.fromarray(img.astype(np.uint8))


def psnr(y, t, max):
    xp = cuda.get_array_module(y)
    mse = xp.mean(xp.square(y - t))
    psnr = 20 * xp.log10(max / xp.sqrt(mse))
    return psnr


def clipped_psnr(y, t, max=1.0, clip=(0.0, 1.0)):
    xp = cuda.get_array_module(y)
    y_c = xp.clip(y, clip[0], clip[1])
    t_c = xp.clip(t, clip[0], clip[1])
    mse = xp.mean(xp.square(y_c - t_c))
    psnr = 20 * xp.log10(max / xp.sqrt(mse))
    return psnr
