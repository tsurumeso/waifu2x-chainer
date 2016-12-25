#!/bin/sh

python train.py --gpu 0 --dataset_dir dataset --patches 8 --validation_crops 64 --active_cropping_tries 5 --epoch 10 --model_name reference_scale_rgb --lr_decay_interval 3

read Wait
