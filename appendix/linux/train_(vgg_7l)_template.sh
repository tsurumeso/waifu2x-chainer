#!/bin/sh

python train.py --gpu 0 --dataset_dir dataset --patches 8 --validation_crops 64 --active_cropping_tries 5 --epoch 10 --model_name reference_scale_rgb --lr_decay_interval 3
python train.py --gpu 0 --dataset_dir dataset --finetune models/reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir dataset --method noise --noise_level 0 --finetune models/reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir dataset --method noise --noise_level 1 --finetune models/reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir dataset --method noise --noise_level 2 --finetune models/reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir dataset --method noise --noise_level 3 --finetune models/reference_scale_rgb.npz

read Wait
