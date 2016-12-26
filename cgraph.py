import argparse
import numpy as np
import chainer.computational_graph as c

from lib import utils
from lib import srcnn


p = argparse.ArgumentParser()
p.add_argument('--arch', choices=[
        'VGG_7l',
        'UpConv_7l',
        'SRResNet_10l',
        'ResUpConv_10l'], default='VGG_7l')

args = p.parse_args()

model = srcnn.archs[args.arch](3)
offset = utils.offset_size(model)
x = np.zeros((1, 3, offset + 64, offset + 64), dtype=np.float32)
y = model(x)
g = c.build_computational_graph(y)
with open(args.arch + '_cgraph.dot', 'w') as o:
    o.write(g.dump())
