from __future__ import print_function

import os
import argparse
import numpy as np
import chainer
from chainer import cuda
from PIL import Image

from lib import iproc
from lib import srcnn
from lib import reconstruct


p = argparse.ArgumentParser()
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--src', default='images/small_noisy1.jpg')
p.add_argument('--arch',
               choices=['VGG_7l', '0',
                        'UpConv_7l', '1',
                        'SRResNet_10l', '2',
                        'ResUpConv_10l', '3'],
               default='VGG_7l')
p.add_argument('--scale', action='store_true')
p.add_argument('--noise', action='store_true')
p.add_argument('--noise_level', type=int, choices=[0, 1, 2, 3], default=1)
p.add_argument('--color', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', action='store_true')
p.add_argument('--block_size', type=int, default=64)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--psnr', default='')
p.add_argument('--test', action='store_true')

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]
if args.test:
    args.scale = True
    args.noise = True
    args.noise_level = 1


if __name__ == '__main__':
    ch = 3 if args.color == 'rgb' else 1
    model_dir = 'models/%s' % args.arch.lower()
    if args.scale:
        model_name = '%s/anime_style_scale_%s.npz' % (model_dir, args.color)
        model_scale = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_name, model_scale)
    if args.noise:
        model_name = '%s/anime_style_noise%d_%s.npz' % \
            (model_dir, args.noise_level, args.color)
        model_noise = srcnn.archs[args.arch](ch)
        chainer.serializers.load_npz(model_name, model_noise)

    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
        if args.scale:
            model_scale.to_gpu()
        if args.noise:
            model_noise.to_gpu()

    src = dst = Image.open(args.src)
    icc_profile = src.info.get('icc_profile')

    basename = os.path.basename(args.src)
    output, ext = os.path.splitext(basename)
    output += '_'
    
    denoise_func = reconstruct.noise
    upscale_func = reconstruct.scale
    if args.tta:
        denoise_func = reconstruct.noise_tta
        upscale_func = reconstruct.scale_tta
        output += '(tta)'

    if args.noise:
        print('Level %d denoising...' % args.noise_level, end=' ', flush=True)
        dst = denoise_func(model_noise, dst,
                           args.block_size, args.batch_size)
        output += '(noise%d)' % args.noise_level
        print('OK')
    if args.scale:
        print('2x upscaling...', end=' ', flush=True)
        dst = upscale_func(model_scale, dst,
                           args.block_size, args.batch_size)
        output += '(scale2x)'
        print('OK')

    output += '(%s).png' % args.arch.lower()
    dst.save(output, icc_profile=icc_profile)
    print('Output saved as \'%s\'' % output)

    if not args.psnr == '':
        original = iproc.read_image_rgb_uint8(args.psnr)
        print('PSNR: ' + str(iproc.psnr(original, np.array(dst), 255.)) + ' dB')
