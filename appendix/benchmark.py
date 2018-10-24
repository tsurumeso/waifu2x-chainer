from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
import time

import chainer
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
from PIL import Image
import six

sys.path.append('..')
from lib import iproc  # NOQA
from lib import pairwise_transform  # NOQA
from lib import reconstruct  # NOQA
from lib import srcnn  # NOQA
from lib import utils  # NOQA


def denoise_image(cfg, src, model):
    dst = src.copy()
    six.print_('Level {} denoising...'.format(cfg.noise_level),
               end=' ', flush=True)
    if cfg.tta:
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    six.print_('OK')
    return dst


def upscale_image(cfg, src, model):
    dst = src.copy()
    six.print_('2.0x upscaling...', end=' ', flush=True)
    if model.inner_scale == 1:
        dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
    if cfg.tta:
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
    six.print_('OK')
    return dst


def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1
    model_dir = '../models/{}'.format(cfg.arch.lower())

    models = {}
    if cfg.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
        else:
            model_name = 'anime_style_noise{}_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)
            models['noise'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise'])
            model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            model_path = os.path.join(model_dir, model_name)
            models['scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'scale':
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])

    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models


def benchmark(models, images, sampling_factor, quality):
    scores = []
    for src in images:
        dst = pairwise_transform.scale(
            np.array(src), [args.downsampling_filter], 1, 1, False)
        if quality != 100 or args.method != 'scale':
            with iproc.array_to_wand(dst) as tmp:
                tmp = iproc.jpeg(tmp, sampling_factor, quality)
                dst = iproc.wand_to_array(tmp)
        dst = Image.fromarray(dst)
        if 'noise_scale' in models:
            dst = upscale_image(args, dst, models['noise_scale'])
        else:
            if 'noise' in models:
                dst = denoise_image(args, dst, models['noise'])
            if 'scale' in models:
                dst = upscale_image(args, dst, models['scale'])
        score = iproc.clipped_psnr(
            np.array(dst), np.array(src), a_max=255)
        scores.append(score)
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores))


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--input', '-i', default='../images/original.png')
p.add_argument('--arch', '-a', default='')
p.add_argument('--method', '-m', choices=['scale', 'noise_scale'],
               default='scale')
p.add_argument('--noise_level', '-n', type=int, choices=[0, 1],
               default=1)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', '-t', action='store_true')
p.add_argument('--tta_level', '-T', type=int, choices=[2, 4, 8], default=8)
p.add_argument('--batch_size', '-b', type=int, default=16)
p.add_argument('--block_size', '-l', type=int, default=128)
p.add_argument('--chroma_subsampling', '-j', action='store_true')
p.add_argument('--downsampling_filter', '-d', default='box')

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]


if __name__ == '__main__':
    utils.set_random_seed(0, args.gpu)

    if os.path.isdir(args.input):
        filelist = utils.load_filelist(args.input)
    else:
        filelist = [args.input]

    images = []
    for path in filelist:
        basename = os.path.basename(path)
        _, ext = os.path.splitext(basename)
        if ext.lower() in ['.png', '.bmp', '.tif', '.tiff']:
            img = Image.open(path).convert('RGB')
            w, h = img.size[:2]
            img = img.crop((0, 0, w - (w % 2), h - (h % 2)))
            images.append(img)

    qualities = [50, 60, 70, 80, 90, 100]
    sampling_factor = '1x1,1x1,1x1'
    if args.chroma_subsampling:
        sampling_factor = '2x2,1x1,1x1'

    arch_scores = {}
    for arch in srcnn.table.values():
        args.arch = arch
        models = load_models(args)
        scores = []
        sems = []
        for quality in qualities:
            print(arch, quality)
            start = time.time()
            score, sem = benchmark(models, images, sampling_factor, quality)
            scores.append(score)
            sems.append(sem)
            print('Elapsed time: {:.6f} sec'.format(time.time() - start))
        arch_scores[arch] = [scores, sems]

    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    title = 'scale ({}) '.format(args.downsampling_filter)
    title += 'noise{} ({}) '.format(
        args.noise_level, sampling_factor) if 'noise' in args.method else ''
    title += 'tta{}'.format(args.tta_level) if args.tta else ''
    plt.title(title)
    plt.xlabel('JPEG quality')
    plt.ylabel('PSNR [dB]')
    plt.ylim(27.5, 42)
    if args.method == 'scale':
        plt.xticks([50, 60, 70, 80, 90, 100], [50, 60, 70, 80, 90, 'lossless'])
    else:
        plt.xticks([50, 60, 70, 80, 90, 100])
    plt.yticks([30, 35, 40])
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(2.5))
    if args.method == 'noise_scale':
        if args.noise_level == 0:
            plt.axvspan(85, 100, color='b', alpha=0.1, lw=0)
        elif args.noise_level == 1:
            plt.axvspan(65, 90, color='b', alpha=0.1, lw=0)
    plt.grid(which='both', color='gray', linestyle='--')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    for key, value in arch_scores.items():
        plt.errorbar(qualities, value[0], yerr=value[1],
                     fmt='o-', capsize=3, label=key)
    plt.legend(loc='upper left', edgecolor='white')
    plt.show()
