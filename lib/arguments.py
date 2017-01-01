import os
import argparse
import numpy as np


class Namespace():

    def __init__(self, kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def append(self, key, value):
        setattr(self, key, value)


p = argparse.ArgumentParser(
               description='Chainer implementation of waifu2x model trainer')
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--dataset_dir', required=True)
p.add_argument('--validation_rate', type=float, default=0.05)
p.add_argument('--color', choices=['y', 'rgb'], default='rgb')
p.add_argument('--arch',
               choices=['VGG_7l',
                        'UpConv_7l',
                        'SRResNet_10l',
                        'ResUpConv_10l'],
               default='VGG_7l')
p.add_argument('--method', choices=['noise', 'scale', 'noise_scale'],
               default='scale')
p.add_argument('--noise_level', type=int, choices=[0, 1, 2, 3], default=1)
p.add_argument('--nr_rate', type=float, default=0.75)
p.add_argument('--chroma_subsampling_rate', type=float, default=0.0)
p.add_argument('--crop_size', type=int, default=64)
p.add_argument('--max_size', type=int, default=256)
p.add_argument('--active_cropping_rate', type=float, default=0.5)
p.add_argument('--active_cropping_tries', type=int, default=10)
p.add_argument('--random_half_rate', type=float, default=0.0)
p.add_argument('--random_unsharp_mask_rate', type=float, default=0.0)
p.add_argument('--learning_rate', type=float, default=0.0005)
p.add_argument('--lr_min', type=float, default=0.00001)
p.add_argument('--lr_decay', type=float, default=0.9)
p.add_argument('--lr_decay_interval', type=int, default=5)
p.add_argument('--batch_size', type=int, default=8)
p.add_argument('--patches', type=int, default=16)
p.add_argument('--validation_crops', type=int, default=128)
p.add_argument('--resize_blur_min', type=float, default=0.95)
p.add_argument('--resize_blur_max', type=float, default=1.05)
p.add_argument('--epoch', type=int, default=100)
p.add_argument('--inner_epoch', type=int, default=4)
p.add_argument('--finetune', default=None)
p.add_argument('--model_name', default=None)
p.add_argument('--test', action='store_true')
p.add_argument('--test_dir', default='./test')

args = Namespace(vars(p.parse_args()))

if args.test:
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

if args.color == 'y':
    ch, weight = 1, (1.0)
elif args.color == 'rgb':
    ch, weight = 3, (0.29891 * 3, 0.58661 * 3, 0.11448 * 3)
weight = np.array(weight, dtype=np.float32).reshape(1, ch, 1, 1)
args.append('ch', ch)
args.append('weight', weight)
