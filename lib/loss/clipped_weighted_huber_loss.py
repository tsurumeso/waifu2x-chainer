import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ClippedWeightedHuberLoss(function.Function):

    def __init__(self, weight, delta=0.1, clip=(0.0, 1.0)):
        self.weight = weight
        self.delta = delta
        self.clip = clip

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        x0_c = xp.clip(x0, self.clip[0], self.clip[1])
        x1_c = xp.clip(x1, self.clip[0], self.clip[1])
        self.diff = (x0_c - x1_c) * self.weight

        abs_diff = xp.abs(self.diff)
        y = xp.square(abs_diff)
        abs_diff -= abs_diff.dtype.type(self.delta)
        xp.maximum(abs_diff, 0, dtype=abs_diff.dtype, out=abs_diff)
        xp.square(abs_diff, out=abs_diff)
        y = (y - abs_diff) * 0.5

        return y.mean(),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        gy, = grad_outputs
        delta = float(self.delta)
        gx = gy * xp.clip(self.diff, -delta, delta)

        return gx, -gx


def clipped_weighted_huber_loss(x, t, weight, delta=0.1, clip=(0.0, 1.0)):
    return ClippedWeightedHuberLoss(
        weight=weight, delta=delta, clip=clip)(x, t)
