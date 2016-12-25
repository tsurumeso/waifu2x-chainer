#!/bin/sh

python train.py --gpu 0 --dataset_dir dataset --method noise --noise_level 3 --finetune models/reference_scale_rgb.pkl

read Wait
