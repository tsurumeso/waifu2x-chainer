# waifu2x-chainer

Implementation of waifu2x-model-trainer using Chainer. Note that the training procedure of waifu2x-chainer can be slightly different from original waifu2x.

## Requirements

Platform:
  - Python 2.7.6+
  
Python packages:
  - Chainer 1.18.0+
  - Pillow 3.0.0+
  - Wand 0.4.0+
  
Additional dependencies:
  - ImageMagick
  
## Installation

### Install Python packages
```
pip install chainer
pip install pillow
pip install wand
```

### Install additional dependencies:
```
sudo apt-get install ImageMagick
```

### Getting waifu2x-chainer
```
git clone https://github.com/tsurumeso/waifu2x-chainer.git
```

### Testing reconstruction
```
python waifu2x.py --test
```

## References

- [2] nagadomi, "Image Super-Resolution for Anime-Style Art", https://github.com/nagadomi/waifu2x
- [3] "For Creators", http://piapro.net/en_for_creators.html
