import argparse

from lib import srcnn


p = argparse.ArgumentParser(
    description='Chainer implementation of waifu2x model trainer')
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--dataset_dir', required=True)
p.add_argument('--validation_rate', type=float, default=0.05)
p.add_argument('--color', choices=['y', 'rgb'], default='rgb')
p.add_argument('--arch',
               choices=['VGG7l', '0',
                        'UpConv7l', '1',
                        'SRResNet10l', '2',
                        'ResUpConv10l', '3'],
               default='VGG7l')
p.add_argument('--method', choices=['noise', 'scale', 'noise_scale'],
               default='scale')
p.add_argument('--noise_level', type=int, choices=[0, 1, 2, 3], default=1)
p.add_argument('--nr_rate', type=float, default=0.65)
p.add_argument('--chroma_subsampling_rate', type=float, default=0.5)
p.add_argument('--crop_size', type=int, default=64)
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
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--patches', type=int, default=64)
p.add_argument('--validation_crop_rate', type=float, default=0.5)
p.add_argument('--downsampling_filters', nargs='+', default=['box'])
p.add_argument('--resize_blur_min', type=float, default=0.95)
p.add_argument('--resize_blur_max', type=float, default=1.05)
p.add_argument('--epoch', type=int, default=50)
p.add_argument('--inner_epoch', type=int, default=4)
p.add_argument('--finetune', default=None)
p.add_argument('--model_name', default=None)

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]
