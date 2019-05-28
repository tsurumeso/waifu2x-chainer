from __future__ import division
from __future__ import print_function
import argparse
import os
import shutil
import time
import warnings

import chainer
from chainer import optimizers
import numpy as np
import six

from lib import iproc
from lib import srcnn
from lib import utils

from lib.dataset_sampler import DatasetSampler
from lib.loss import clipped_weighted_huber_loss


def train_inner_epoch(model, weight, optimizer, data_queue, batch_size):
    sum_loss = 0
    xp = model.xp
    train_x, train_y = data_queue.get()
    perm = np.random.permutation(len(train_x))
    for i in six.moves.range(0, len(train_x), batch_size):
        local_perm = perm[i:i + batch_size]
        batch_x = xp.array(train_x[local_perm], dtype=np.float32) / 255
        batch_y = xp.array(train_y[local_perm], dtype=np.float32) / 255
        model.cleargrads()
        pred = model(batch_x)
        # loss = F.mean_squared_error(pred, batch_y)
        loss = clipped_weighted_huber_loss(pred, batch_y, weight)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_x)
    return sum_loss / len(train_x)


def valid_inner_epoch(model, data_queue, batch_size):
    sum_score = 0
    xp = model.xp
    valid_x, valid_y = data_queue.get()
    perm = np.random.permutation(len(valid_x))
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for i in six.moves.range(0, len(valid_x), batch_size):
            local_perm = perm[i:i + batch_size]
            batch_x = xp.array(valid_x[local_perm], dtype=np.float32) / 255
            batch_y = xp.array(valid_y[local_perm], dtype=np.float32) / 255
            pred = model(batch_x)
            score = iproc.clipped_psnr(pred.data, batch_y)
            sum_score += float(score) * len(batch_x)
    return sum_score / len(valid_x)


p = argparse.ArgumentParser(description='Chainer implementation of waifu2x')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--seed', '-s', type=int, default=11)
p.add_argument('--dataset_dir', '-d', required=True)
p.add_argument('--validation_rate', type=float, default=0.05)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--arch', '-a',
               choices=['VGG7', '0', 'UpConv7', '1',
                        'ResNet10', '2', 'UpResNet10', '3'],
               default='VGG7')
p.add_argument('--method', '-m', choices=['noise', 'scale', 'noise_scale'],
               default='scale')
p.add_argument('--noise_level', '-n', type=int, choices=[0, 1, 2, 3],
               default=1)
p.add_argument('--nr_rate', type=float, default=0.65)
p.add_argument('--chroma_subsampling_rate', type=float, default=0.5)
p.add_argument('--reduce_memory_usage', action='store_true')
p.add_argument('--out_size', type=int, default=64)
p.add_argument('--max_size', type=int, default=256)
p.add_argument('--active_cropping_rate', type=float, default=0.5)
p.add_argument('--active_cropping_tries', type=int, default=10)
p.add_argument('--random_half_rate', type=float, default=0.0)
p.add_argument('--random_color_noise_rate', type=float, default=0.0)
p.add_argument('--random_unsharp_mask_rate', type=float, default=0.0)
p.add_argument('--learning_rate', type=float, default=0.00025)
p.add_argument('--lr_min', type=float, default=0.00001)
p.add_argument('--lr_decay', type=float, default=0.9)
p.add_argument('--lr_decay_interval', type=int, default=5)
p.add_argument('--batch_size', '-b', type=int, default=16)
p.add_argument('--patches', '-p', type=int, default=64)
p.add_argument('--validation_crop_rate', type=float, default=0.5)
p.add_argument('--downsampling_filters', nargs='+', default=['box'])
p.add_argument('--resize_blur_min', type=float, default=0.95)
p.add_argument('--resize_blur_max', type=float, default=1.05)
p.add_argument('--epoch', '-e', type=int, default=50)
p.add_argument('--inner_epoch', type=int, default=4)
p.add_argument('--finetune', '-f', default=None)
p.add_argument('--model_name', default=None)

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]


warnings.filterwarnings('ignore')
if __name__ == '__main__':
    utils.set_random_seed(args.seed, args.gpu)
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

    print('* setup model...', end=' ')
    if args.model_name is None:
        if args.method == 'noise':
            model_name = 'anime_style_noise{}'.format(args.noise_level)
        elif args.method == 'scale':
            model_name = 'anime_style_scale'
        elif args.method == 'noise_scale':
            model_name = 'anime_style_noise{}_scale'.format(args.noise_level)
        model_path = '{}_{}.npz'.format(model_name, args.color)
    else:
        model_name = args.model_name.rstrip('.npz')
        model_path = model_name + '.npz'
    if not os.path.exists('epoch'):
        os.makedirs('epoch')

    model = srcnn.archs[args.arch](ch)
    if model.offset % model.inner_scale != 0:
        raise ValueError('offset %% inner_scale must be 0.')
    elif model.inner_scale != 1 and model.inner_scale % 2 != 0:
        raise ValueError('inner_scale must be 1 or an even number.')
    if args.finetune is not None:
        chainer.serializers.load_npz(args.finetune, model)

    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        weight = chainer.backends.cuda.to_gpu(weight)
        model.to_gpu()

    optimizer = optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    print('done')

    valid_config = utils.get_config(args, model, train=False)
    train_config = utils.get_config(args, model, train=True)

    print('* check forward path...', end=' ')
    di = train_config.in_size
    do = train_config.out_size
    dx = model.xp.zeros((args.batch_size, 3, di, di), dtype=np.float32)
    dy = model(dx)
    if dy.shape[2:] != (do, do):
        raise ValueError('Invlid output size\n'
                         'Expect: {}\n'
                         'Actual: ({}, {})'.format(dy.shape[2:], do, do))
    print('done')

    print('* starting processes of dataset sampler...', end=' ')
    valid_queue = DatasetSampler(valid_list, valid_config)
    train_queue = DatasetSampler(train_list, train_config)
    print('done')

    best_count = 0
    best_score = 0
    best_loss = np.inf
    for epoch in range(0, args.epoch):
        print('### epoch: {} ###'.format(epoch))
        train_queue.reload_switch(init=(epoch < args.epoch - 1))
        for inner_epoch in range(0, args.inner_epoch):
            best_count += 1
            print('  # inner epoch: {}'.format(inner_epoch))
            start = time.time()
            train_loss = train_inner_epoch(
                model, weight, optimizer, train_queue, args.batch_size)
            if args.reduce_memory_usage:
                train_queue.wait()
            if train_loss < best_loss:
                best_loss = train_loss
                print('    * best loss on training dataset: {:.6f}'.format(
                    train_loss))
            valid_score = valid_inner_epoch(
                model, valid_queue, args.batch_size)
            if valid_score > best_score:
                best_count = 0
                best_score = valid_score
                print('    * best score on validation dataset: PSNR {:.6f} dB'
                      .format(valid_score))
                best_model = model.copy().to_cpu()
                epoch_path = 'epoch/{}_epoch{}.npz'.format(model_name, epoch)
                chainer.serializers.save_npz(model_path, best_model)
                shutil.copy(model_path, epoch_path)
            if best_count >= args.lr_decay_interval:
                best_count = 0
                optimizer.alpha *= args.lr_decay
                if optimizer.alpha < args.lr_min:
                    optimizer.alpha = args.lr_min
                else:
                    print('    * learning rate decay: {:.6f}'.format(
                        optimizer.alpha))
            print('    * elapsed time: {:.6f} sec'.format(time.time() - start))
