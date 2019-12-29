from __future__ import print_function
import os
import sys

import chainer
from chainer.exporters import caffe
import numpy as np
import onnx_chainer

sys.path.append('..')
from lib import srcnn  # NOQA


def rename_caffe_model(dir, filename):
    model_path = os.path.join(dir, 'chainer_model.caffemodel')
    prototxt_path = os.path.join(dir, 'chainer_model.prototxt')
    new_model_path = os.path.join(dir, filename + '.caffemodel')
    new_prototxt_path = os.path.join(dir, filename + '.prototxt')
    os.rename(model_path, new_model_path)
    os.rename(prototxt_path, new_prototxt_path)


def main():
    for key, value in srcnn.archs.items():
        model_dir = '../models/{}'.format(key.lower())
        for filename in os.listdir(model_dir):
            basename, ext = os.path.splitext(filename)
            onnx_path = os.path.join(model_dir, basename + '.onnx')
            if ext == '.npz':
                model_path = os.path.join(model_dir, filename)
                print(model_path)
                channels = 3 if 'rgb' in filename else 1
                model = value(channels)
                size = 64 + model.offset
                data = np.zeros((1, channels, size, size), dtype=np.float32)
                x = chainer.Variable(data)
                try:
                    chainer.serializers.load_npz(model_path, model)
                    caffe.export(model, [x], model_dir, True, basename)
                    rename_caffe_model(model_dir, basename)
                except Exception:
                    print('Skipped caffe model export')
                onnx_chainer.export(model, x, filename=onnx_path)


if __name__ == '__main__':
    main()
