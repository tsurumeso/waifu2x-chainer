import argparse

import chainer.computational_graph as c
import numpy as np

from lib import srcnn
from lib import utils


p = argparse.ArgumentParser()
p.add_argument('--arch',
               choices=['VGG7', '0',
                        'UpConv7', '1',
                        'ResNet10', '2',
                        'ResUpConv10', '3'],
               default='VGG7')

args = p.parse_args()
if args.arch in srcnn.table:
    args.arch = srcnn.table[args.arch]

if __name__ == '__main__':
    model = srcnn.archs[args.arch](3)
    offset = utils.offset_size(model)
    x = np.zeros((1, 3, offset + 64, offset + 64), dtype=np.float32)
    y = model(x)
    g = c.build_computational_graph(y)
    with open(args.arch + '_cgraph.dot', 'w') as o:
        o.write(g.dump())
