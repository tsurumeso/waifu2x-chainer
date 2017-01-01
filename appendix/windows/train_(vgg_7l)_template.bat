cd ../../
python train.py --gpu 0 --dataset_dir jpeg_dataset --patches 8 --validation_crops 64 --active_cropping_tries 5 --epoch 10 --model_name reference_scale_rgb --lr_decay_interval 3
python train.py --gpu 0 --dataset_dir png_dataset --finetune reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir png_dataset --method noise --noise_level 0 --finetune reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir png_dataset --method noise --noise_level 1 --finetune reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir png_dataset --method noise --noise_level 2 --finetune reference_scale_rgb.npz
python train.py --gpu 0 --dataset_dir png_dataset --method noise --noise_level 3 --finetune reference_scale_rgb.npz
pause
