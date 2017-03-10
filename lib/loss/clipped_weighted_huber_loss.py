from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy


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
        y = xp.square(self.diff)
        mask = y > (self.delta ** 2)
        y -= mask * xp.square(abs(self.diff) - self.delta)
        y *= 0.5
        return xp.array(y.sum() / y.dtype.type(y.size), dtype=y.dtype),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        mask = xp.abs(self.diff) <= self.delta
        # In loss function, gy[0] is initialized to ones array
        coeff = gy[0] * gy[0].dtype.type(1. / self.diff.size)
        gx = coeff * xp.where(mask, self.diff, self.delta * xp.sign(self.diff))
        return gx, -gx


def clipped_weighted_huber_loss(x, t, weight, delta=0.1, clip=(0.0, 1.0)):
    return ClippedWeightedHuberLoss(
        weight=weight, delta=delta, clip=clip)(x, t)
