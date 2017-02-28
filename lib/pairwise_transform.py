from __future__ import division

import random
import numpy as np
from PIL import Image

from lib import iproc
from lib import data_augmentation


def _noise(src, level, chroma):
    # YUV 444
    sampling_factor = '1x1,1x1,1x1'
    if np.random.uniform() < chroma:
        # YUV 420
        sampling_factor = '2x2,1x1,1x1'
    if level == 0:
        dst = iproc.jpeg(src, sampling_factor, random.randint(85, 100))
    elif level == 1:
        dst = iproc.jpeg(src, sampling_factor, random.randint(65, 90))
    elif level == 2 or level == 3:
        # for level 3, --nr_rate 1
        rand = np.random.uniform()
        if rand < 0.6:
            quality = random.randint(25, 70)
            dst = iproc.jpeg(src, sampling_factor, quality)
        elif rand < 0.9:
            quality0 = random.randint(35, 70)
            quality1 = random.randint(25, 65)
            dst = iproc.jpeg(src, sampling_factor, quality0)
            dst = iproc.jpeg(dst, sampling_factor, quality1)
        else:
            quality0 = random.randint(50, 70)
            quality1 = random.randint(35, 65)
            quality2 = random.randint(25, 55)
            dst = iproc.jpeg(src, sampling_factor, quality0)
            dst = iproc.jpeg(dst, sampling_factor, quality1)
            dst = iproc.jpeg(dst, sampling_factor, quality2)
    return dst


def noise(src, rate, level, chroma):
    if np.random.uniform() < rate:
        with iproc.array_to_wand(src) as tmp:
            tmp = _noise(tmp, level, chroma)
            dst = iproc.wand_to_array(tmp)
        return dst
    else:
        return src


def scale(src, downsampling_filters, bmin, bmax):
    # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
    # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
    # 'lanczos2', 'sinc'
    h, w = src.shape[:2]
    blur = np.random.uniform(bmin, bmax)
    rand = random.randint(0, len(downsampling_filters)-1)
    with iproc.array_to_wand(src) as tmp:
        tmp.resize(w // 2, h // 2, downsampling_filters[rand], blur)
        tmp.resize(w, h, 'box')
        dst = iproc.wand_to_array(tmp)
    return dst


def noise_scale(src, downsampling_filters, bmin, bmax, rate, level, chroma):
    # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
    # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
    # 'lanczos2', 'sinc'
    h, w = src.shape[:2]
    blur = np.random.uniform(bmin, bmax)
    rand = random.randint(0, len(downsampling_filters)-1)
    with iproc.array_to_wand(src) as tmp:
        tmp.resize(w // 2, h // 2, downsampling_filters[rand], blur)
        if np.random.uniform() < rate:
            tmp = _noise(tmp, level, chroma)
        tmp.resize(w, h, 'box')
        dst = iproc.wand_to_array(tmp)
    return dst


def crop_if_large(src, max_size):
    if max_size > 0 and src.shape[1] > max_size and src.shape[0] > max_size:
        point_x = random.randint(0, src.shape[1] - max_size)
        point_y = random.randint(0, src.shape[0] - max_size)
        dst = src[point_y:point_y + max_size, point_x:point_x + max_size, :]
        return dst
    return src


def preprocess(src, cfg):
    dst = data_augmentation.half(src, cfg.random_half_rate)
    dst = crop_if_large(dst, cfg.max_size)
    dst = data_augmentation.flip(dst)
    dst = data_augmentation.color_noise(dst, cfg.random_color_noise_rate)
    dst = data_augmentation.unsharp_mask(dst, cfg.random_unsharp_mask_rate)
    dst = data_augmentation.shift_1px(dst)
    return dst


def active_cropping(x, y, size, p, tries):
    if np.random.uniform() < p:
        best_mse = 0
        best_cx = np.zeros((size, size, x.shape[2]), dtype=np.uint8)
        best_cy = np.zeros((size, size, y.shape[2]), dtype=np.uint8)
        for i in range(tries):
            point_x = random.randint(0, x.shape[1] - size)
            point_y = random.randint(0, x.shape[0] - size)
            crop_x = x[point_y:point_y + size, point_x:point_x + size, :]
            crop_y = y[point_y:point_y + size, point_x:point_x + size, :]
            mse = np.mean(np.square(crop_y - crop_x))
            if mse >= best_mse:
                best_mse = mse
                best_cx = crop_x
                best_cy = crop_y
        return best_cx, best_cy
    else:
        point_x = random.randint(0, x.shape[1] - size)
        point_y = random.randint(0, x.shape[0] - size)
        crop_x = x[point_y:point_y + size, point_x:point_x + size, :]
        crop_y = y[point_y:point_y + size, point_x:point_x + size, :]
        return crop_x, crop_y


def pairwise_transform(src, cfg):
    unstable_region_offset = 8
    top = cfg.offset // 2
    bottom = cfg.insize - top
    y = preprocess(src, cfg)

    if cfg.method == 'scale':
        x = scale(
            y, cfg.downsampling_filters,
            cfg.resize_blur_min, cfg.resize_blur_max)
    elif cfg.method == 'noise':
        x = noise(y, cfg.nr_rate, cfg.noise_level, cfg.chroma_subsampling_rate)
    elif cfg.method == 'noise_scale':
        x = noise_scale(
            y, cfg.downsampling_filters,
            cfg.resize_blur_min, cfg.resize_blur_max,
            cfg.nr_rate, cfg.noise_level, cfg.chroma_subsampling_rate)

    y = y[unstable_region_offset:y.shape[0] - unstable_region_offset,
          unstable_region_offset:y.shape[1] - unstable_region_offset]
    x = x[unstable_region_offset:x.shape[0] - unstable_region_offset,
          unstable_region_offset:x.shape[1] - unstable_region_offset]

    patch_x = np.zeros(
        (cfg.patches, cfg.ch, cfg.insize, cfg.insize), dtype=np.uint8)
    patch_y = np.zeros(
        (cfg.patches, cfg.ch, cfg.crop_size, cfg.crop_size), dtype=np.uint8)

    for i in range(cfg.patches):
        crop_x, crop_y = active_cropping(
            x, y, cfg.insize,
            cfg.active_cropping_rate, cfg.active_cropping_tries)
        if cfg.ch == 1:
            ycbcr_x = Image.fromarray(crop_x).convert('YCbCr')
            ycbcr_y = Image.fromarray(crop_y).convert('YCbCr')
            crop_x = np.array(ycbcr_x)[:, :, 0]
            crop_y = np.array(ycbcr_y)[top:bottom, top:bottom, 0]
            patch_x[i] = crop_x.reshape(cfg.ch, cfg.insize, cfg.insize)
            patch_y[i] = crop_y.reshape(cfg.ch, cfg.crop_size, cfg.crop_size)
        elif cfg.ch == 3:
            crop_y = crop_y[top:bottom, top:bottom, :]
            crop_x = crop_x.transpose(2, 0, 1)
            crop_y = crop_y.transpose(2, 0, 1)
            patch_x[i] = crop_x.reshape(cfg.ch, cfg.insize, cfg.insize)
            patch_y[i] = crop_y.reshape(cfg.ch, cfg.crop_size, cfg.crop_size)
    return patch_x, patch_y
