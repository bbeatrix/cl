from collections import OrderedDict
from functools import partial

from absl import app, logging
import gin
import torch
import torch.nn as nn
import torchvision as tv
from torchsummary import summary


@gin.configurable(blacklist=['device', 'input_shape', 'output_shape'])
class ModelBuilder():
    def __init__(self, device, input_shape, output_shape, model_class):
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_class = model_class

        self._build()

    def _build(self):
        self.model = self.model_class(self.input_shape,
                                      self.output_shape)
        self.model.to(self.device)
        logging.info("Model summary")
        summary(self.model, self.input_shape)


@gin.configurable
def resnet18(input_shape, output_shape):
    return ResNet(input_shape, output_shape, block=ResNetBasicBlock, depths=[2, 2, 2, 2])

@gin.configurable
def resnet34(input_shape, output_shape):
    return ResNet(input_shape, output_shape, block=ResNetBasicBlock, depths=[3, 4, 6, 3])

@gin.configurable
def resnet50(input_shape, output_shape):
    return ResNet(input_shape, output_shape, block=ResNetBottleNeckBlock, depths=[3, 4, 6, 3])


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3,
                 *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        layers = OrderedDict({'conv' : nn.Conv2d(self.in_channels,
                                                 self.expanded_channels,
                                                 kernel_size=1,
                                                 stride=self.downsampling,
                                                 bias=False),
                              'bn' : nn.BatchNorm2d(self.expanded_channels)})
        self.shortcut = nn.Sequential(layers) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    layers = OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) })
    return nn.Sequential(layers)


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        first_conv_bn = conv_bn(self.in_channels,
                                self.out_channels,
                                conv=self.conv,
                                bias=False,
                                stride=self.downsampling)
        second_conv_bn = conv_bn(self.out_channels,
                                 self.expanded_channels,
                                 conv=self.conv,
                                 bias=False)
        self.blocks = nn.Sequential(first_conv_bn,
                                    activation(),
                                    second_conv_bn)


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        first_conv_bn = conv_bn(self.in_channels,
                                self.out_channels,
                                self.conv,
                                kernel_size=1)
        second_conv_bn = conv_bn(self.out_channels,
                                 self.out_channels,
                                 self.conv,
                                 kernel_size=3,
                                 stride=self.downsampling)
        third_conv_bn = conv_bn(self.out_channels,
                                self.expanded_channels,
                                self.conv,
                                kernel_size=1)

        self.blocks = nn.Sequential(first_conv_bn,
                                    activation(),
                                    second_conv_bn,
                                    activation(),
                                    third_conv_bn)


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()

        downsampling = 2 if in_channels != out_channels else 1

        first_block = block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling)
        following_blocks = []
        for _ in range(n - 1):
            next_block = block(out_channels * block.expansion,
                               out_channels,
                               downsampling=1,
                               *args,
                               **kwargs)
            following_blocks.append(next_block)

        self.blocks = nn.Sequential(first_block, *following_blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetFeatures(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        gate_conv_layer = nn.Conv2d(in_channels,
                                    self.blocks_sizes[0],
                                    kernel_size=7,
                                    stride=2,
                                    padding=3,
                                    bias=False)
        self.gate = nn.Sequential(gate_conv_layer,
                                  nn.BatchNorm2d(self.blocks_sizes[0]),
                                  activation(),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        first_layer = ResNetLayer(blocks_sizes[0],
                                  blocks_sizes[0],
                                  n=depths[0],
                                  activation=activation,
                                  block=block,
                                  *args,
                                  **kwargs)
        following_layers = []
        for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:]):
            next_layer = ResNetLayer(in_channels * block.expansion,
                                     out_channels,
                                     n=n,
                                     activation=activation,
                                     block=block,
                                     *args,
                                     **kwargs)
            following_layers.append(next_layer)

        self.blocks = nn.ModuleList([first_layer, *following_layers])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetTop(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super().__init__()
        self.features = ResNetFeatures(input_shape[0], *args, **kwargs)
        self.predictions = ResnetTop(self.features.blocks[-1].blocks[-1].expanded_channels,
                                     output_shape)

    def forward(self, x):
        x = self.features(x)
        out = self.predictions(x)
        return out


def main(argv):
    model = resnet18((3, 32, 32), 100)
    logging.info("Model summary")
    summary(model.cuda(), (3, 32, 32))
    logging.info("C'est un magnifique modèle!")


if __name__ == '__main__':
    app.run(main)
