#!/bin/sh

python train.py --gpu 0 --dataset_dir dataset --finetune models/reference_scale_rgb.pkl

read Wait
