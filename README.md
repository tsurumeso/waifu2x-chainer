# waifu2x-chainer

Chainer implementation of waifu2x [[1]](https://github.com/nagadomi/waifu2x) and its model trainer.
Note that the training procedure of waifu2x-chainer can be slightly different from original waifu2x.

## Summery

<img src="https://raw.githubusercontent.com/tsurumeso/waifu2x-chainer/master/images/summery.png" width="886">

- 2D character picture (Kagamine Rin) is licensed under CC BY-NC by piapro [2].

## Requirements

### Platform
  - Python 2.7.6+, 3.5.1+, 3.6.0+

### Python packages
  - Chainer
  - Cupy (for GPU support)
  - Pillow
  - Wand (for training)

## Installation

### Install Python packages
```
pip install chainer
pip install cupy
pip install pillow
pip install wand
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

### Noise reduction
```
python waifu2x.py --method noise --noise_level 1 --input path/to/image/or/directory --arch VGG7

python waifu2x.py -m noise -n 0 -i path/to/image/or/directory -a 0
python waifu2x.py -m noise -n 2 -i path/to/image/or/directory -a 0
python waifu2x.py -m noise -n 3 -i path/to/image/or/directory -a 0
```

### 2x upscaling
```
python waifu2x.py --method scale --input path/to/image/or/directory --arch VGG7

python waifu2x.py -m scale -i path/to/image/or/directory -a 0
```

### Noise reduction + 2x upscaling
```
python waifu2x.py --method noise_scale --noise_level 1 --input path/to/image/or/directory --arch VGG7

python waifu2x.py -m noise_scale -n 0 -i path/to/image/or/directory -a 0
python waifu2x.py -m noise_scale -n 2 -i path/to/image/or/directory -a 0
python waifu2x.py -m noise_scale -n 3 -i path/to/image/or/directory -a 0
```

### Train your own model

Please see template script at
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/linux">appendix/linux</a>
or
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/windows">appendix/windows</a>
. In my case, 5000 JPEG images are used for pretraining and 1000 noise-free-PNG images for finetuning.

## References

- [1] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
- [2] "For Creators", http://piapro.net/en_for_creators.html
