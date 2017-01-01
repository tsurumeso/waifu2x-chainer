import random
from iproc import *
from PIL import Image, ImageFilter


def unsharp_mask(src, p):
    if np.random.uniform() < p:
        pil = Image.fromarray(src)
        radius = random.randint(1, 3)
        percent = random.randint(10, 90)
        threshold = random.randint(0, 5)
        unsharp = ImageFilter.UnsharpMask(radius=radius,
                                          percent=percent,
                                          threshold=threshold)
        dst = np.array(pil.filter(unsharp))
        return dst.astype(np.float32)
    else:
        return src


def random_flip(src):
    rand = random.randint(0, 3)
    dst = src.copy()
    if rand == 1:
        dst = src[::-1, :, :]
    elif rand == 2:
        dst = src[:, ::-1, :]
    elif rand == 3:
        dst = src[::-1, ::-1, :]
    return dst


def random_half(src, p):
    dst = array_to_wand(src)
    if np.random.uniform() < p:
        # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
        # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
        # 'sinc'
        filter = ('box', 'box', 'blackman', 'cubic', 'lanczos')
        h, w = src.shape[:2]
        rand = random.randint(0, len(filter) - 1)
        dst.resize(w / 2, h / 2, filter[rand])
    return wand_to_array(dst)


def shift_1px(src):
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
