from __future__ import division
import argparse
import os
import time

import chainer
import numpy as np
from PIL import Image
import six

from lib import iproc
from lib import reconstruct
from lib import srcnn
from lib import utils


def denoise_image(cfg, src, model):
    dst, alpha = split_alpha(src, model)
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
    if alpha is not None:
        dst.putalpha(alpha)
    return dst


def upscale_image(cfg, src, scale_model, alpha_model=None):
    dst, alpha = split_alpha(src, scale_model)
    for i in range(int(np.ceil(np.log2(cfg.scale_ratio)))):
        six.print_('2.0x upscaling...', end=' ', flush=True)
        model = scale_model if i == 0 or alpha_model is None else alpha_model
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        if cfg.tta:
            dst = reconstruct.image_tta(
                dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
        else:
            dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
        if alpha_model is None:
            alpha = reconstruct.image(
                alpha, scale_model, cfg.block_size, cfg.batch_size)
        else:
            alpha = reconstruct.image(
                alpha, alpha_model, cfg.block_size, cfg.batch_size)
        six.print_('OK')
    dst_w = int(np.round(src.size[0] * cfg.scale_ratio))
    dst_h = int(np.round(src.size[1] * cfg.scale_ratio))
    if dst_w != dst.size[0] or dst_h != dst.size[1]:
        six.print_('Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        six.print_('OK')
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


def split_alpha(src, model):
    alpha = None
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        six.print_('Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        six.print_('OK')
    return rgb, alpha


def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1
    if cfg.model_dir is None:
        model_dir = 'models/{}'.format(cfg.arch.lower())
    else:
        model_dir = cfg.model_dir

    models = {}
    flag = False
    if cfg.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            alpha_model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            alpha_model_path = os.path.join(model_dir, alpha_model_name)
            models['alpha'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            flag = True
    if cfg.method == 'scale' or flag:
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'noise' or flag:
        model_name = 'anime_style_noise{}_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)
        models['noise'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models


p = argparse.ArgumentParser(description='Chainer implementation of waifu2x')
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--input', '-i', default='images/small.png')
p.add_argument('--output', '-o', default='./')
p.add_argument('--extension', '-e', choices=['png', 'webp'], default='png')
p.add_argument('--quality', '-q', type=int, default=None)
p.add_argument('--arch', '-a',
               choices=['VGG7', '0', 'UpConv7', '1',
                        'ResNet10', '2', 'UpResNet10', '3'],
               default='VGG7')
p.add_argument('--model_dir', '-d', default=None)
p.add_argument('--method', '-m', choices=['noise', 'scale', 'noise_scale'],
               default='scale')
p.add_argument('--scale_ratio', '-s', type=float, default=2.0)
p.add_argument('--noise_level', '-n', type=int, choices=[0, 1, 2, 3],
               default=1)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', '-t', action='store_true')
p.add_argument('--tta_level', '-T', type=int, choices=[2, 4, 8], default=8)
p.add_argument('--batch_size', '-b', type=int, default=16)
p.add_argument('--block_size', '-l', type=int, default=128)
g = p.add_mutually_exclusive_group()
g.add_argument('--width', '-W', type=int, default=0)
g.add_argument('--height', '-H', type=int, default=0)
g.add_argument('--shorter_side', '-S', type=int, default=0)
g.add_argument('--longer_side', '-L', type=int, default=0)

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]


if __name__ == '__main__':
    models = load_models(args)

    input_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']
    output_exts = ['.png', '.webp']
    outext = '.' + args.extension

    outname = None
    outdir = args.output
    if os.path.isdir(args.input):
        filelist = utils.load_filelist(args.input)
    else:
        tmpname, tmpext = os.path.splitext(os.path.basename(args.output))
        if tmpext in output_exts:
            outext = tmpext
            outname = tmpname
            outdir = os.path.dirname(args.output)
            outdir = './' if outdir == '' else outdir
        elif not tmpext == '':
            raise ValueError('Format {} is not supported'.format(tmpext))
        filelist = [args.input]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for path in filelist:
        if outname is None or len(filelist) > 1:
            outname, outext = os.path.splitext(os.path.basename(path))
        outpath = os.path.join(outdir, '{}{}'.format(outname, outext))
        if outext.lower() in input_exts:
            src = Image.open(path)
            w, h = src.size[:2]
            if args.width != 0:
                args.scale_ratio = args.width / w
            elif args.height != 0:
                args.scale_ratio = args.height / h
            elif args.shorter_side != 0:
                if w < h:
                    args.scale_ratio = args.shorter_side / w
                else:
                    args.scale_ratio = args.shorter_side / h
            elif args.longer_side != 0:
                if w > h:
                    args.scale_ratio = args.longer_side / w
                else:
                    args.scale_ratio = args.longer_side / h

            dst = src.copy()
            start = time.time()
            outname += '_(tta{})'.format(args.tta_level) if args.tta else '_'
            if 'noise_scale' in models:
                outname += '(noise{}_scale{:.1f}x)'.format(
                    args.noise_level, args.scale_ratio)
                dst = upscale_image(
                    args, dst, models['noise_scale'], models['alpha'])
            else:
                if 'noise' in models:
                    outname += '(noise{})'.format(args.noise_level)
                    dst = denoise_image(args, dst, models['noise'])
                if 'scale' in models:
                    outname += '(scale{:.1f}x)'.format(args.scale_ratio)
                    dst = upscale_image(args, dst, models['scale'])
            print('Elapsed time: {:.6f} sec'.format(time.time() - start))

            outname += '({}_{}){}'.format(args.arch, args.color, outext)
            if os.path.exists(outpath):
                outpath = os.path.join(outdir, outname)

            lossless = args.quality is None
            quality = 100 if lossless else args.quality
            icc_profile = src.info.get('icc_profile')
            icc_profile = "" if icc_profile is None else icc_profile
            dst.convert(src.mode).save(
                outpath, quality=quality, lossless=lossless,
                icc_profile=icc_profile)
            six.print_('Saved as \'{}\''.format(outpath))
