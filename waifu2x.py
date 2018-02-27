import argparse
import os

import chainer
from chainer import cuda
import numpy as np
from PIL import Image
import six

from lib import iproc
from lib import reconstruct
from lib import srcnn
from lib import utils


def denoise_image(src, model, cfg):
    dst, alpha = split_alpha(src, model.offset)
    six.print_('Level %d denoising...' % cfg.noise_level, end=' ', flush=True)
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


def upscale_image(src, scale_model, cfg, alpha_model=None):
    dst, alpha = split_alpha(src, scale_model.offset)
    log_scale = np.log2(cfg.scale_ratio)
    for i in range(int(np.ceil(log_scale))):
        six.print_('2.0x upscaling...', end=' ', flush=True)
        model = alpha_model
        if i == 0 or alpha_model is None:
            model = scale_model
        if model.inner_scale == 1:
            dst = iproc.scale2x(dst)  # Nearest neighbor 2x scaling
            alpha = iproc.scale2x(alpha)  # Nearest neighbor 2x scaling
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
    if np.round(log_scale % 1.0, 6) != 0 or log_scale <= 0:
        six.print_('Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        six.print_('OK')
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


def split_alpha(src, offset):
    alpha = None
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        six.print_('Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, offset)
        six.print_('OK')
    return rgb, alpha


def load_models(args):
    ch = 3 if args.color == 'rgb' else 1
    if args.model_dir is None:
        model_dir = 'models/%s' % args.arch.lower()
    else:
        model_dir = args.model_dir

    models = {}
    flag = False
    if args.method == 'noise_scale':
        model_name = ('anime_style_noise%d_scale_%s.npz'
                      % (args.noise_level, args.color))
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[args.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            alpha_model_name = 'anime_style_scale_%s.npz' % args.color
            alpha_model_path = os.path.join(model_dir, alpha_model_name)
            models['alpha'] = srcnn.archs[args.arch](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            flag = True
    if args.method == 'scale' or flag:
        model_name = 'anime_style_scale_%s.npz' % args.color
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if args.method == 'noise' or flag:
        model_name = ('anime_style_noise%d_%s.npz'
                      % (args.noise_level, args.color))
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_name = ('anime_style_noise%d_scale_%s.npz'
                          % (args.noise_level, args.color))
            model_path = os.path.join(model_dir, model_name)
        models['noise'] = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--input', '-i', default='images/small.png')
p.add_argument('--output', '-o', default='./')
p.add_argument('--arch', '-a',
               choices=['VGG7', '0',
                        'UpConv7', '1',
                        'ResNet10', '2'],
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
p.add_argument('--batch_size', '-b', type=int, default=8)
p.add_argument('--block_size', '-l', type=int, default=64)
p.add_argument('--width', '-W', type=int, default=0)
p.add_argument('--height', '-H', type=int, default=0)

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]
if args.width != 0 and args.height != 0:
    args.height = 0


if __name__ == '__main__':
    models = load_models(args)

    if '.png' not in args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
    else:
        dirname = os.path.dirname(args.output)
        if len(dirname) == 0:
            dirname = './'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    if os.path.isdir(args.input):
        filelist = utils.load_filelist(args.input)
    else:
        filelist = [args.input]

    for path in filelist:
        src = Image.open(path)
        w, h = src.size[:2]
        if args.width != 0:
            args.scale_ratio = args.width / w
        if args.height != 0:
            args.scale_ratio = args.height / h
        basename = os.path.basename(path)
        outname, ext = os.path.splitext(basename)
        if ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            outname += ('_(tta%d)' % args.tta_level if args.tta else '_')
            dst = src.copy()
            if 'noise_scale' in models:
                outname += ('(noise%d_scale%.1fx)'
                            % (args.noise_level, args.scale_ratio))
                dst = upscale_image(
                    dst, models['noise_scale'], args, models['alpha'])
            else:
                if 'noise' in models:
                    outname += '(noise%d)' % args.noise_level
                    dst = denoise_image(dst, models['noise'], args)
                if 'scale' in models:
                    outname += '(scale%.1fx)' % args.scale_ratio
                    dst = upscale_image(dst, models['scale'], args)

            if args.model_dir is None:
                outname += '(%s_%s).png' % (args.arch.lower(), args.color)
            else:
                outname += '(model_%s).png' % args.color

            if os.path.isdir(args.output):
                outpath = os.path.join(args.output, outname)
            else:
                if os.path.exists(args.output):
                    outpath = os.path.join(dirname, outname)
                else:
                    outpath = args.output
            dst.convert(src.mode).save(
                outpath, icc_profile=src.info.get('icc_profile'))
            six.print_('Saved as \'%s\'' % outpath)
