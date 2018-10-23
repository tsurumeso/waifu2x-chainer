import chainer
import chainer.functions as F
import chainer.links as L


class VGG7(chainer.Chain):

    def __init__(self, ch):
        super(VGG7, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(ch, 32, 3)
            self.conv2 = L.Convolution2D(32, 32, 3)
            self.conv3 = L.Convolution2D(32, 64, 3)
            self.conv4 = L.Convolution2D(64, 64, 3)
            self.conv5 = L.Convolution2D(64, 128, 3)
            self.conv6 = L.Convolution2D(128, 128, 3)
            self.conv7 = L.Convolution2D(128, ch, 3)

        self.ch = ch
        self.offset = 7
        self.inner_scale = 1

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x), 0.1)
        h = F.leaky_relu(self.conv2(h), 0.1)
        h = F.leaky_relu(self.conv3(h), 0.1)
        h = F.leaky_relu(self.conv4(h), 0.1)
        h = F.leaky_relu(self.conv5(h), 0.1)
        h = F.leaky_relu(self.conv6(h), 0.1)
        h = self.conv7(h)
        return h


class UpConv7(chainer.Chain):

    def __init__(self, ch):
        super(UpConv7, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(ch, 16, 3)
            self.conv2 = L.Convolution2D(16, 32, 3)
            self.conv3 = L.Convolution2D(32, 64, 3)
            self.conv4 = L.Convolution2D(64, 128, 3)
            self.conv5 = L.Convolution2D(128, 256, 3)
            self.conv6 = L.Convolution2D(256, 256, 3)
            self.conv7 = L.Deconvolution2D(256, ch, 4, 2, 3, nobias=True)

        self.ch = ch
        self.offset = 14
        self.inner_scale = 2

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x), 0.1)
        h = F.leaky_relu(self.conv2(h), 0.1)
        h = F.leaky_relu(self.conv3(h), 0.1)
        h = F.leaky_relu(self.conv4(h), 0.1)
        h = F.leaky_relu(self.conv5(h), 0.1)
        h = F.leaky_relu(self.conv6(h), 0.1)
        h = self.conv7(h)
        return h


class ResBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, slope=0.1):
        super(ResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, 3)
            self.conv2 = L.Convolution2D(out_channels, out_channels, 3)

        if in_channels != out_channels:
            conv_bridge = L.Convolution2D(in_channels, out_channels, 1)
            self.add_link('conv_bridge', conv_bridge)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slope = slope

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x), self.slope)
        h = F.leaky_relu(self.conv2(h), self.slope)
        if self.in_channels != self.out_channels:
            x = self.conv_bridge(x[:, :, 2:-2, 2:-2])
        else:
            x = x[:, :, 2:-2, 2:-2]
        return h + x


class ResNet10(chainer.Chain):

    def __init__(self, ch):
        super(ResNet10, self).__init__()
        with self.init_scope():
            self.conv_pre = L.Convolution2D(ch, 64, 3)
            self.res1 = ResBlock(64, 64)
            self.res2 = ResBlock(64, 64)
            self.res3 = ResBlock(64, 64)
            self.res4 = ResBlock(64, 64)
            self.res5 = ResBlock(64, 64)
            self.conv_bridge = L.Convolution2D(64, 64, 3)
            self.conv_post = L.Convolution2D(64, ch, 3)

        self.ch = ch
        self.offset = 13
        self.inner_scale = 1

    def __call__(self, x):
        h = skip = F.leaky_relu(self.conv_pre(x), 0.1)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.leaky_relu(self.conv_bridge(h), 0.1)
        h = h + skip[:, :, 11:-11, 11:-11]
        h = self.conv_post(h)
        return h


class SEResBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, r=16, slope=0.1):
        super(SEResBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, 3)
            self.conv2 = L.Convolution2D(out_channels, out_channels, 3)
            self.fc1 = L.Linear(out_channels, out_channels // r)
            self.fc2 = L.Linear(out_channels // r, out_channels)

        if in_channels != out_channels:
            conv_bridge = L.Convolution2D(in_channels, out_channels, 1)
            self.add_link('conv_bridge', conv_bridge)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slope = slope

    def __call__(self, x):
        h = F.leaky_relu(self.conv1(x), self.slope)
        h = F.leaky_relu(self.conv2(h), self.slope)
        se = F.relu(self.fc1(F.average(h, axis=(2, 3))))
        se = F.sigmoid(self.fc2(se))[:, :, None, None]
        se = F.broadcast_to(se, h.shape)
        if self.in_channels != self.out_channels:
            x = self.conv_bridge(x[:, :, 2:-2, 2:-2])
        else:
            x = x[:, :, 2:-2, 2:-2]
        return h * se + x


class UpResNet10(chainer.Chain):

    def __init__(self, ch):
        super(UpResNet10, self).__init__()
        with self.init_scope():
            self.conv_pre = L.Convolution2D(ch, 64, 3)
            self.res1 = SEResBlock(64, 64, 4)
            self.res2 = SEResBlock(64, 64, 4)
            self.res3 = SEResBlock(64, 64, 4)
            self.res4 = SEResBlock(64, 64, 4)
            self.res5 = SEResBlock(64, 64, 4)
            self.conv_bridge = L.Convolution2D(64, 64, 3)
            self.conv_post = L.Deconvolution2D(64, ch, 4, 2, 3, nobias=True)

        self.ch = ch
        self.offset = 26
        self.inner_scale = 2

    def __call__(self, x):
        h = skip = F.leaky_relu(self.conv_pre(x), 0.1)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.leaky_relu(self.conv_bridge(h), 0.1)
        h = h + skip[:, :, 11:-11, 11:-11]
        h = self.conv_post(h)
        return h


archs = {
    'VGG7': VGG7,
    'UpConv7': UpConv7,
    'ResNet10': ResNet10,
    'UpResNet10': UpResNet10,
}

table = {
    '0': 'VGG7',
    '1': 'UpConv7',
    '2': 'ResNet10',
    '3': 'UpResNet10',
}
