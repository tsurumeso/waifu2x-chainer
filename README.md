# waifu2x-chainer

Chainer implementation of waifu2x [[1]](https://github.com/nagadomi/waifu2x) and its model trainer.
Note that the training procedure of waifu2x-chainer may be slightly different from original waifu2x.

## Summary

![](images/summery.png)

- 2D character picture (Kagamine Rin) is licensed under CC BY-NC by piapro [2].

## Requirements

  - Chainer
  - Cupy (for GPU support)
  - Matplotlib (for benchmark)
  - ONNX-Chainer (for ONNX model export)
  - Pillow
  - Wand (for training)

## Installation

### Install Python packages
```
pip install chainer
pip install cupy
pip install pillow
```

### Getting waifu2x-chainer
```
git clone https://github.com/tsurumeso/waifu2x-chainer.git
```

### Testing
```
cd waifu2x-chainer
python waifu2x.py
```

## Usage

Omitting --gpu (-g) option run on CPU.

### Noise reduction
```
python waifu2x.py --method noise --noise_level 1 --input path/to/image/or/directory --arch VGG7 --gpu 0

python waifu2x.py -m noise -n 0 -i path/to/image/or/directory -a 0 -g 0
python waifu2x.py -m noise -n 2 -i path/to/image/or/directory -a 0 -g 0
python waifu2x.py -m noise -n 3 -i path/to/image/or/directory -a 0 -g 0
```

### 2x upscaling
```
python waifu2x.py --method scale --input path/to/image/or/directory --arch VGG7 --gpu 0

python waifu2x.py -m scale -i path/to/image/or/directory -a 0 -g 0
```

### Noise reduction + 2x upscaling
```
python waifu2x.py --method noise_scale --noise_level 1 --input path/to/image/or/directory --arch VGG7 --gpu 0

python waifu2x.py -m noise_scale -n 0 -i path/to/image/or/directory -a 0 -g 0
python waifu2x.py -m noise_scale -n 2 -i path/to/image/or/directory -a 0 -g 0
python waifu2x.py -m noise_scale -n 3 -i path/to/image/or/directory -a 0 -g 0
```

## Train your own model

### Install Wand
```
pip install wand
```

For more details, please refer template training script at
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/linux">appendix/linux</a>
or
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/windows">appendix/windows</a>
. In my case, 5000 JPEG images are used for pretraining and 1000 noise-free-PNG images for finetuning.

## Convert Chainer models to ONNX and Caffe models

### Install ONNX-Chainer
```
pip install onnx-chainer
```

### Run script
```
cd appendix
python convert_models.py
```

Converted models are saved at the same directory of the original models
(e.g. models/vgg7/anime_style_scale_rgb.npz to models/vgg7/anime_style_scale_rgb.caffemodel).
Since Crop Layer is currently not supported on Chainer and ONNX, ResNet10 model export cannot be done.

## References

- [1] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
- [2] "For Creators", http://piapro.net/en_for_creators.html
