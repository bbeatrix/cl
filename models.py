import gin
import torch
import torch.nn as nn
import torchvision as tv

from absl import app
from functools import partial
from collections import OrderedDict
from torchsummary import summary


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # dynamic add padding based on the kernel_size
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

    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        layers = OrderedDict({'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                                 stride=self.downsampling, bias=False),
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
        first_conv_bn = conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling)
        second_conv_bn = conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
        self.blocks = nn.Sequential(first_conv_bn,
                                    activation(),
                                    second_conv_bn)


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        first_conv_bn = conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1)
        second_conv_bn = conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling)
        third_conv_bn = conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)

        self.blocks = nn.Sequential(first_conv_bn,
                                    activation(),
                                    second_conv_bn,
                                    activation(),
                                    third_conv_bn)


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        first_block = block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling)
        following_blocks = [block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]

        self.blocks = nn.Sequential(first_block, *following_blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetFeatures(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        first_layer = ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                                  block=block,  *args, **kwargs)
        following_layers = [ResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs)
                            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]

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


@gin.configurable
class ResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.features = ResNetFeatures(in_channels, *args, **kwargs)
        self.predictions = ResnetTop(self.features.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.features(x)
        out = self.predictions(x)
        return out


@gin.configurable
def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[2, 2, 2, 2])


def resnet34(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, deepths=[3, 4, 6, 3])


def resnet50(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock, deepths=[3, 4, 6, 3])


def main(argv):
    #model = tv.models.resnet50(pretrained=True)
    model = resnet18(3, 100)
    summary(model.cuda(), (3, 32, 32))

    print("C'est magnifique!")


if __name__ == '__main__':
    app.run(main)
