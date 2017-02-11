import os
import six
import glob
import argparse
import numpy as np
import chainer
from chainer import cuda
from PIL import Image

from lib import iproc
from lib import srcnn
from lib import reconstruct


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
    iter = 0
    while iter < int(np.ceil(cfg.scale_factor / 2)):
        iter += 1
        six.print_('2.0x upscaling...', end=' ', flush=True)
        if cfg.tta:
            dst = reconstruct.image_tta(
                src, model, True,
                cfg.tta_level, cfg.block_size, cfg.batch_size)
        else:
            dst = reconstruct.image(
                src, model, True, cfg.block_size, cfg.batch_size)
        six.print_('OK')
    if np.round(cfg.scale_factor % 2.0, 6) != 0:
        six.print_('resizing...', end=' ', flush=True)
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
               choices=['VGG_7l', '0',
                        'UpConv_7l', '1',
                        'SRResNet_10l', '2',
                        'ResUpConv_10l', '3'],
               default='VGG_7l')
p.add_argument('--scale', '-s', action='store_true')
p.add_argument('--scale_factor', type=float, default=2.0)
p.add_argument('--noise', '-n', action='store_true')
p.add_argument('--noise_level', type=int, choices=[0, 1, 2, 3], default=1)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', action='store_true')
p.add_argument('--tta_level', type=int, choices=[2, 4, 8], default=8)
p.add_argument('--block_size', type=int, default=64)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--test', '-t', action='store_true')

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
    model_dir = 'models/%s' % args.arch.lower()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.scale:
        model_name = '%s/anime_style_scale_%s.npz' % (model_dir, args.color)
        model_scale = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_name, model_scale)

    if args.noise:
        model_name = ('%s/anime_style_noise%d_%s.npz'
                      % (model_dir, args.noise_level, args.color))
        model_noise = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_name, model_noise)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        if args.scale:
            model_scale.to_gpu()
        if args.noise:
            model_noise.to_gpu()

    if os.path.isdir(args.input):
        args.input = args.input.replace('\\', '/')
        filelist = glob.glob(args.input.strip('/') + '/*.*')
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
            if args.noise:
                oname += '(noise%d)' % args.noise_level
                dst = denoise_image(dst, model_noise, args)
            if args.scale:
                oname += '(scale%.1fx)' % args.scale_factor
                dst = upscale_image(dst, model_scale, args)

            oname += '(%s).png' % args.arch.lower()
            opath = os.path.join(args.output, oname)
            dst.save(opath, icc_profile=icc_profile)
            six.print_('Saved as \'%s\'' % opath)
