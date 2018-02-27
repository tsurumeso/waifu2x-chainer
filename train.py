from __future__ import division
from __future__ import print_function
import os
import shutil
import time
import warnings

import chainer
from chainer import cuda
from chainer import optimizers
import numpy as np
import six

from lib import iproc
from lib import srcnn
from lib import utils

from lib.dataset_sampler import DatasetSampler
from lib.loss import clipped_weighted_huber_loss
from lib.settings import args


def train_inner_epoch(model, weight, optimizer, data_queue, batch_size):
    sum_loss = 0
    scale = 1. / 255.
    xp = model.xp
    train_x, train_y = data_queue.get()
    perm = np.random.permutation(len(train_x))
    for i in six.moves.range(0, len(train_x), batch_size):
        local_perm = perm[i:i + batch_size]
        batch_x = xp.array(train_x[local_perm], dtype=np.float32) * scale
        batch_y = xp.array(train_y[local_perm], dtype=np.float32) * scale
        model.cleargrads()
        pred = model(batch_x)
        # loss = F.mean_squared_error(pred, batch_y)
        loss = clipped_weighted_huber_loss(pred, batch_y, weight)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data * len(batch_x)
    return sum_loss / len(train_x)


def valid_inner_epoch(model, data_queue, batch_size):
    sum_score = 0
    scale = 1. / 255.
    xp = model.xp
    valid_x, valid_y = data_queue.get()
    perm = np.random.permutation(len(valid_x))
    for i in six.moves.range(0, len(valid_x), batch_size):
        local_perm = perm[i:i + batch_size]
        batch_x = xp.array(valid_x[local_perm], dtype=np.float32) * scale
        batch_y = xp.array(valid_y[local_perm], dtype=np.float32) * scale
        pred = model(batch_x)
        score = iproc.clipped_psnr(pred.data, batch_y)
        sum_score += score * len(batch_x)
    return sum_score / len(valid_x)


def get_config(base_args, model, train=True):
    ch = model.ch
    offset = model.offset
    inner_scale = model.inner_scale
    crop_size = base_args.out_size + offset * 2
    in_size = crop_size // inner_scale

    if train:
        max_size = base_args.max_size
        patches = base_args.patches
    else:
        max_size = 0
        coeff = (1 - base_args.validation_rate) / base_args.validation_rate
        patches = int(
            round(base_args.validation_crop_rate * coeff * base_args.patches))

    config = {
        'ch': ch,
        'method': base_args.method,
        'noise_level': base_args.noise_level,
        'nr_rate': base_args.nr_rate,
        'chroma_subsampling_rate': base_args.chroma_subsampling_rate,
        'offset': offset,
        'crop_size': crop_size,
        'in_size': in_size,
        'out_size': base_args.out_size,
        'inner_scale': inner_scale,
        'max_size': max_size,
        'active_cropping_rate': base_args.active_cropping_rate,
        'active_cropping_tries': base_args.active_cropping_tries,
        'random_half_rate': base_args.random_half_rate,
        'random_color_noise_rate': base_args.random_color_noise_rate,
        'random_unsharp_mask_rate': base_args.random_unsharp_mask_rate,
        'patches': patches,
        'downsampling_filters': base_args.downsampling_filters,
        'resize_blur_min': base_args.resize_blur_min,
        'resize_blur_max': base_args.resize_blur_max,
    }
    return utils.Namespace(config)


def train():
    if args.color == 'y':
        ch = 1
        weight = (1.0,)
    elif args.color == 'rgb':
        ch = 3
        weight = (0.29891 * 3, 0.58661 * 3, 0.11448 * 3)
    weight = np.array(weight, dtype=np.float32)
    weight = weight[:, np.newaxis, np.newaxis]

    print('* loading filelist...', end=' ')
    filelist = utils.load_filelist(args.dataset_dir, shuffle=True)
    valid_num = int(np.ceil(args.validation_rate * len(filelist)))
    valid_list, train_list = filelist[:valid_num], filelist[valid_num:]
    print('done')

    print('* loading model...', end=' ')
    if args.model_name is None:
        if args.method == 'noise':
            model_name = ('anime_style_noise%d_%s'
                          % (args.noise_level, args.color))
        elif args.method == 'scale':
            model_name = 'anime_style_scale_%s' % args.color
        elif args.method == 'noise_scale':
            model_name = ('anime_style_noise%d_scale_%s'
                          % (args.noise_level, args.color))
    else:
        model_name = args.model_name.rstrip('.npz')
    model_path = model_name + '.npz'
    if not os.path.exists('epoch'):
        os.makedirs('epoch')

    model = srcnn.archs[args.arch](ch)
    if args.finetune is not None:
        chainer.serializers.load_npz(args.finetune, model)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        weight = cuda.cupy.array(weight)
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    print('done')

    valid_config = get_config(args, model, train=False)
    train_config = get_config(args, model, train=True)

    print('* starting processes of dataset sampler...', end=' ')
    valid_queue = DatasetSampler(valid_list, valid_config)
    train_queue = DatasetSampler(train_list, train_config)
    print('done')

    best_count = 0
    best_score = 0
    best_loss = np.inf
    for epoch in range(0, args.epoch):
        print('### epoch: %d ###' % epoch)
        train_queue.reload_switch(init=(epoch < args.epoch - 1))
        for inner_epoch in range(0, args.inner_epoch):
            best_count += 1
            print('  # inner epoch: %d' % inner_epoch)
            start = time.time()
            train_loss = train_inner_epoch(
                model, weight, optimizer, train_queue, args.batch_size)
            train_queue.wait()
            if train_loss < best_loss:
                best_loss = train_loss
                print('    * best loss on train dataset: %f' % train_loss)
            valid_score = valid_inner_epoch(
                model, valid_queue, args.batch_size)
            if valid_score > best_score:
                best_count = 0
                best_score = valid_score
                print('    * best score on validation dataset: PSNR %f dB'
                      % valid_score)
                best_model = model.copy().to_cpu()
                epoch_path = 'epoch/%s_epoch%d.npz' % (model_name, epoch)
                chainer.serializers.save_npz(model_path, best_model)
                shutil.copy(model_path, epoch_path)
            if best_count >= args.lr_decay_interval:
                best_count = 0
                optimizer.alpha *= args.lr_decay
                if optimizer.alpha < args.lr_min:
                    optimizer.alpha = args.lr_min
                else:
                    print('    * learning rate decay: %f' % optimizer.alpha)
            print('    * elapsed time: %f sec' % (time.time() - start))


warnings.filterwarnings('ignore')
if __name__ == '__main__':
    utils.set_random_seed(args.seed, args.gpu)
    train()
