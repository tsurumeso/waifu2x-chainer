import argparse
import glob
import os

import chainer
from chainer import cuda
import numpy as np
from PIL import Image
import six

from lib import reconstruct
from lib import srcnn


def denoise_image(src, model, cfg):
    six.print_('Level %d denoising...' % cfg.noise_level, end=' ', flush=True)
    if cfg.tta:
        dst = reconstruct.image_tta(
            src, model, False,
            cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(
            src, model, False, cfg.block_size, cfg.batch_size)
    six.print_('OK')
    return dst


def upscale_image(src, model, cfg):
    dst = src
    log_scale = np.log2(cfg.scale_factor)
    upsampling = (model.inner_scale == 1)
    for _ in range(int(np.ceil(log_scale))):
        six.print_('2.0x upscaling...', end=' ', flush=True)
        if cfg.tta:
            dst = reconstruct.image_tta(
                dst, model, upsampling,
                cfg.tta_level, cfg.block_size, cfg.batch_size)
        else:
            dst = reconstruct.image(
                dst, model, upsampling, cfg.block_size, cfg.batch_size)
        six.print_('OK')
    if np.round(log_scale % 1.0, 6) != 0:
        six.print_('Resizing...', end=' ', flush=True)
        dst_w = int(np.round(src.size[0] * cfg.scale_factor))
        dst_h = int(np.round(src.size[1] * cfg.scale_factor))
        dst = dst.resize((dst_w, dst_h), Image.ANTIALIAS)
        six.print_('OK')
    return dst


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--input', '-i', default='images/small_noisy1.jpg')
p.add_argument('--output', '-o', default='./')
p.add_argument('--arch', '-a',
               choices=['VGG7', '0',
                        'UpConv7', '1',
                        'ResNet10', '2',
                        'ResUpConv10', '3'],
               default='VGG7')
p.add_argument('--model_dir', '-m', default=None)
p.add_argument('--scale', '-s', action='store_true')
p.add_argument('--scale_factor', '-S', type=float, default=2.0)
p.add_argument('--noise', '-n', action='store_true')
p.add_argument('--noise_scale', action='store_true')
p.add_argument('--noise_level', '-N', type=int, choices=[0, 1, 2, 3],
               default=1)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', '-t', action='store_true')
p.add_argument('--tta_level', '-T', type=int, choices=[2, 4, 8], default=8)
p.add_argument('--block_size', type=int, default=64)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--test', action='store_true')

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]
if args.test:
    args.scale = True
    args.noise = True
    args.noise_level = 1
formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

if __name__ == '__main__':
    ch = 3 if args.color == 'rgb' else 1
    if args.model_dir is None:
        model_dir = 'models/%s' % args.arch.lower()
    else:
        model_dir = args.model_dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.noise_scale:
        model_name = ('anime_style_noise%d_scale_%s.npz'
                      % (args.noise_level, args.color))
        model_path = os.path.join(model_dir, model_name)
        model_noise_scale = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_path, model_noise_scale)
    else:
        if args.scale:
            model_name = 'anime_style_scale_%s.npz' % args.color
            model_path = os.path.join(model_dir, model_name)
            model_scale = srcnn.archs[args.arch](ch)
            chainer.serializers.load_npz(model_path, model_scale)
        if args.noise:
            model_name = ('anime_style_noise%d_%s.npz'
                          % (args.noise_level, args.color))
            model_path = os.path.join(model_dir, model_name)
            model_noise = srcnn.archs[args.arch](ch)
            chainer.serializers.load_npz(model_path, model_noise)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        if args.scale:
            model_scale.to_gpu()
        if args.noise:
            model_noise.to_gpu()

    if os.path.isdir(args.input):
        filelist = glob.glob(os.path.join(args.input, '*.*'))
    else:
        filelist = [args.input]

    for path in filelist:
        src = Image.open(path)
        icc_profile = src.info.get('icc_profile')
        basename = os.path.basename(path)
        oname, ext = os.path.splitext(basename)
        if ext.lower() in formats:
            oname += ('_(tta%d)' % args.tta_level if args.tta else '_')

            dst = src.copy()
            if args.noise_scale:
                oname += '(noise%d_scale)' % args.noise_level
                dst = upscale_image(dst, model_noise_scale, args)
            else:
                if args.noise:
                    oname += '(noise%d)' % args.noise_level
                    dst = denoise_image(dst, model_noise, args)
                if args.scale:
                    oname += '(scale%.1fx)' % args.scale_factor
                    dst = upscale_image(dst, model_scale, args)

            if args.model_dir is None:
                oname += '(%s_%s).png' % (args.arch.lower(), args.color)
            else:
                oname += '(model_%s).png' % args.color
            opath = os.path.join(args.output, oname)
            dst.save(opath, icc_profile=icc_profile)
            six.print_('Saved as \'%s\'' % opath)
