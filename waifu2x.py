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
               choices=['VGG_7l',
                        'UpConv_7l',
                        'SRResNet_10l',
                        'ResUpConv_10l'],
               default='VGG_7l')
p.add_argument('--scale', action='store_true')
p.add_argument('--noise', action='store_true')
p.add_argument('--noise_level', type=int, choices=[0, 1, 2, 3], default=1)
p.add_argument('--color', choices=['y', 'rgb'], default='rgb')
p.add_argument('--block_size', type=int, default=64)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--psnr', default='')
p.add_argument('--test', action='store_true')
args = p.parse_args()

if args.test:
    args.scale = True
    args.noise = True
    args.noise_level = 1

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
basename = os.path.basename(args.src)
output, ext = os.path.splitext(basename)
output += '_'
if args.noise:
    print 'Level %d denoising...' % args.noise_level,
    dst = reconstruct.noise(model_noise, dst, args.block_size, args.batch_size)
    output += '(noise%d)' % args.noise_level
    print 'OK'
if args.scale:
    print '2x upsampling...',
    dst = reconstruct.scale(model_scale, dst, args.block_size, args.batch_size)
    output += '(scale2x)'
    print 'OK'

output += '(%s).png' % args.arch.lower()
dst.save(output)
print 'Output saved as \'%s\'' % output

if not args.psnr == '':
    original = iproc.read_image_rgb_uint8(args.psnr)
    print 'PSNR: ' + str(iproc.psnr(original, np.array(dst), 255.)) + ' dB'
