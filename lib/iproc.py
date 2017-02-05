from __future__ import division

import six
import numpy as np
from chainer import cuda
from PIL import Image
from wand.image import Image as WandImage


def read_image_rgb_uint8(path):
    img = Image.open(path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    return img


def array_to_wand(src):
    buf = six.BytesIO()
    image = Image.fromarray(src)
    image.save(buf, 'BMP')
    dst = WandImage(blob=bytes(buf.getvalue()))
    return dst


def wand_to_array(src):
    image_str = six.BytesIO(src.make_blob())
    image = Image.open(image_str).convert('RGB')
    dst = np.array(image, dtype=np.uint8)
    return dst


def jpeg(src, sampling_factor, quality):
    src.format = 'jpg'
    src.compression_quality = quality
    src.options['jpeg:sampling-factor'] = sampling_factor
    return WandImage(blob=src.make_blob())


def inv(rot, flip=False):
    if flip:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))[:, ::-1, :]
    else:
        return lambda x: np.rot90(x, rot // 90, axes=(0, 1))


def to_image(data, ch):
    image = cuda.to_cpu(data)
    image = np.clip(image, 0, 1) * 255
    if ch == 1:
        return Image.fromarray(image[0].astype(np.uint8))
    elif ch == 3:
        image = image.transpose(1, 2, 0)
        return Image.fromarray(image.astype(np.uint8))


def psnr(y, t, max):
    xp = cuda.get_array_module(y)
    mse = xp.mean(xp.square(y - t))
    y = 20 * xp.log10(max / xp.sqrt(mse))
    return y


def clipped_psnr(y, t, max=1.0, clip=(0.0, 1.0)):
    xp = cuda.get_array_module(y)
    y = xp.clip(y, clip[0], clip[1])
    t = xp.clip(t, clip[0], clip[1])
    mse = xp.mean(xp.square(y - t))
    y = 20 * xp.log10(max / xp.sqrt(mse))
    return y
