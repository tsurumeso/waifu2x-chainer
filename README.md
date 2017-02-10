# waifu2x-chainer

Chainer implementation of waifu2x[[1]](https://github.com/nagadomi/waifu2x) and its model trainer. Note that the training procedure of waifu2x-chainer can be slightly different from original waifu2x.

## Summery

<img src="https://raw.githubusercontent.com/tsurumeso/waifu2x-chainer/master/images/summery.png" width="886">

## Requirements

#### Platform
  - Python 2.7.6+, 3.5.1+
  
#### Python packages
  - Chainer 1.18.0+
  - Pillow 3.0.0+
  - Wand 0.4.0+
  
#### Additional dependencies
  - ImageMagick
  
## Installation

#### Install Python packages
```
pip install chainer
pip install pillow
pip install wand
```

#### Install additional dependencies
```
sudo apt-get install ImageMagick
```

#### Getting waifu2x-chainer
```
git clone https://github.com/tsurumeso/waifu2x-chainer.git
```

#### Testing
```
python waifu2x.py --test
```

## Usage

#### Noise reduction
```
python waifu2x.py --noise --noise_level 1 --input <image or directory> --arch VGG_7l
```

#### 2x upscaling
```
python waifu2x.py --scale --input <image or directory> --arch VGG_7l
```

#### Noise reduction + 2x upscaling
```
python waifu2x.py --scale --noise --noise_level 1 --input <image or directory> --arch VGG_7l
```

#### Train model

Please see template script at
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/linux">appendix/linux</a>
or
<a href="https://github.com/tsurumeso/waifu2x-chainer/tree/master/appendix/windows">appendix/windows</a>

## References

- [1] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
- [2] "For Creators", http://piapro.net/en_for_creators.html
