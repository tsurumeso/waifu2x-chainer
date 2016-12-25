import numpy as np
from PIL import Image
from chainer import cuda

from lib import utils


def blockwise(model, src, block_size, batch_size):
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
    h, w, ch = src.shape
    offset = utils.offset_size(model)
    xp = utils.get_model_module(model)
    src = src.transpose(2, 0, 1) / 255.

    ph = block_size - (h % block_size) + offset / 2
    pw = block_size - (w % block_size) + offset / 2
    src = np.array([np.pad(x, ((offset / 2, ph), (offset / 2, pw)), 'edge')
        for x in src])
    nh = src.shape[1] / block_size
    nw = src.shape[2] / block_size

    x = np.ndarray((nh * nw, ch, block_size + offset, block_size + offset), dtype=np.float32)    
    for i in range(0, nh):
        ih = i * block_size
        for j in range(0, nw):
            index = (i * nh) + j
            iw = j * block_size
            src_ij = src[:, ih:ih + block_size + offset, iw:iw + block_size + offset]
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
            index = (i * nh) + j
            iw = j * block_size
            dst[:, ih:ih + block_size, iw:iw + block_size] = y[index]

    dst = dst[:, :h, :w]
    dst = dst.transpose(1, 2, 0)
    return np.clip(dst, 0, 1) * 255
    
    
def image(model, src):
    if src.ndim == 2:
        src = src[:, :, np.newaxis]
    h, w, ch = src.shape
    xp = utils.get_model_module(model)
    offset = utils.offset_size(model)
    src = src.transpose(2, 0, 1)
    x = np.array([np.pad(c, offset / 2, 'edge') for c in src])
    x = xp.array(x).reshape(1, ch, h, w)
    x = x.astype(np.float32) / 255.
    return model(x)


def scale(model, src, block_size, batch_size):
    src = src.resize((src.size[0] * 2, src.size[1] * 2), Image.NEAREST)
    if model.ch == 1:
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(model, src[:, :, 0], block_size, batch_size)
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr').convert('RGB')
        return dst
    elif model.ch == 3:
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        dst = blockwise(model, src, block_size, batch_size)
        dst = Image.fromarray(dst.astype(np.uint8))
        return dst


def noise(model, src, block_size, batch_size):
    if model.ch == 1:
        src = np.array(src.convert('YCbCr'), dtype=np.uint8)
        dst = blockwise(model, src[:, :, 0], block_size, batch_size)
        src[:, :, 0] = dst[:, :, 0]
        dst = Image.fromarray(src, mode='YCbCr').convert('RGB')
        return dst
    elif model.ch == 3:
        src = np.array(src.convert('RGB'), dtype=np.uint8)
        dst = blockwise(model, src, block_size, batch_size)
        dst = Image.fromarray(dst.astype(np.uint8))
        return dst
