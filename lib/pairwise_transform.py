from data_augmentation import *


def _noise(src, level, chroma):
    sampling_factor = '1x1,1x1,1x1'
    quality0 = random.randint(85, 100)
    quality1 = random.randint(65, 90)
    quality2 = random.randint(45, 70)
    quality3 = random.randint(25, 50)
    if np.random.uniform() < chroma:
        sampling_factor = '2x1,1x1,1x1'
    if level == 0:
        dst = jpeg(src, sampling_factor, quality0)
    elif level == 1:
        dst = jpeg(src, sampling_factor, quality1)
    elif level == 2:
        if np.random.uniform() > 0.5:
            src = jpeg(src, sampling_factor, quality1)
        dst = jpeg(src, sampling_factor, quality2)
    elif level == 3:
        if np.random.uniform() > 0.5:
            src = jpeg(src, sampling_factor, quality1)
        if np.random.uniform() > 0.5:
            src = jpeg(src, sampling_factor, quality2)
        dst = jpeg(src, sampling_factor, quality3)
    return dst


def noise(src, rate, level, chroma):
    if np.random.uniform() < rate:
        src = array_to_wand(src)
        dst = _noise(src, level, chroma)
        return wand_to_array(dst)
    else:
        return src


def scale(src, bmin, bmax):
    # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
    # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
    # 'sinc'
    h, w = src.shape[:2]
    filters = ('box', 'lanczos')
    blur = np.random.uniform(bmin, bmax)
    rand = random.randint(0, len(filters)-1)
    dst = array_to_wand(src)
    dst.resize(w / 2, h / 2, filters[rand], blur)
    dst.resize(w, h, 'box')
    return wand_to_array(dst)


def noise_scale(src, bmin, bmax, rate, level, chroma):
    # 'box', 'triangle', 'hermite', 'hanning', 'hamming', 'blackman',
    # 'gaussian', 'quadratic', 'cubic', 'catrom', 'mitchell', 'lanczos',
    # 'sinc'
    h, w = src.shape[:2]
    filters = ('box', 'lanczos')
    blur = np.random.uniform(bmin, bmax)
    rand = random.randint(0, len(filters)-1)
    dst = array_to_wand(src)
    dst.resize(w / 2, h / 2, filters[rand], blur)
    if np.random.uniform() < rate:
        dst = _noise(dst, level, chroma)
    dst.resize(w, h, 'box')
    return wand_to_array(dst)


def crop_if_large(src, max_size):
    if max_size > 0 and src.shape[1] > max_size and src.shape[0] > max_size:
        point_x = random.randint(0, src.shape[1] - max_size)
        point_y = random.randint(0, src.shape[0] - max_size)
        return src[point_y:point_y + max_size, point_x:point_x + max_size, :]
    return src


def preprocess(src, cfg):
    dst = src
    dst = random_half(src, cfg.random_half_rate)
    dst = crop_if_large(dst, cfg.max_size)
    dst = random_flip(dst)
    dst = unsharp_mask(dst, cfg.random_unsharp_mask_rate)
    dst = shift_1px(dst)
    return dst


def active_cropping(x, y, size, p, tries):
    if np.random.uniform() < p:
        best_mse = 0
        best_cx = np.empty((1, 1, 1))
        best_cy = np.empty((1, 1, 1))
        for i in range(tries):
            point_x = random.randint(0, x.shape[1] - size)
            point_y = random.randint(0, x.shape[0] - size)
            crop_x = x[point_y:point_y + size, point_x:point_x + size, :]
            crop_y = y[point_y:point_y + size, point_x:point_x + size, :]
            mse = np.mean((crop_y - crop_x) ** 2)
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


def pairwise_transform(src, insize, cfg):
    unstable_region_offset = 8
    top = (insize-cfg.crop_size) / 2
    bottom = insize - top
    y = preprocess(src, cfg)

    if cfg.method == 'scale':
        x = scale(y, cfg.resize_blur_min, cfg.resize_blur_max)
    elif cfg.method == 'noise':
        x = noise(y, cfg.nr_rate, cfg.noise_level, cfg.chroma_subsampling_rate)
    elif cfg.method == 'noise_scale':
        x = noise_scale(y, cfg.resize_blur_min, cfg.resize_blur_max,
                        cfg.nr_rate,
                        cfg.noise_level,
                        cfg.chroma_subsampling_rate)

    y = y[unstable_region_offset:y.shape[0] - unstable_region_offset,
          unstable_region_offset:y.shape[1] - unstable_region_offset]
    x = x[unstable_region_offset:x.shape[0] - unstable_region_offset,
          unstable_region_offset:x.shape[1] - unstable_region_offset]

    patch_x = np.ndarray((cfg.patches, cfg.ch, insize, insize),
                         dtype=np.float32)
    patch_y = np.ndarray((cfg.patches, cfg.ch, cfg.crop_size, cfg.crop_size),
                         dtype=np.float32)

    for i in range(cfg.patches):
        crop_x, crop_y = active_cropping(x, y, insize,
                                         cfg.active_cropping_rate,
                                         cfg.active_cropping_tries)
        if cfg.ch == 1:
            ycbcr_x = Image.fromarray(crop_x).convert('YCbCr')
            ycbcr_y = Image.fromarray(crop_y).convert('YCbCr')
            crop_x = np.array(ycbcr_x)[:, :, 0]
            crop_y = np.array(ycbcr_y)[top:bottom, top:bottom, 0]
            patch_x[i] = crop_x.reshape(cfg.ch, insize, insize)
            patch_y[i] = crop_y.reshape(cfg.ch, cfg.crop_size, cfg.crop_size)
        elif cfg.ch == 3:
            crop_y = crop_y[top:bottom, top:bottom, :]
            crop_x = crop_x.transpose(2, 0, 1)
            crop_y = crop_y.transpose(2, 0, 1)
            patch_x[i] = crop_x.reshape(cfg.ch, insize, insize)
            patch_y[i] = crop_y.reshape(cfg.ch, cfg.crop_size, cfg.crop_size)
    return patch_x, patch_y
