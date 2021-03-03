from collections import OrderedDict
from functools import partial

from absl import app
import gin
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision as tv


@gin.configurable(denylist=['device', 'input_shape', 'output_shape'])
class Model:
    def __init__(self, device, input_shape, output_shape, model_path=None, model_class=gin.REQUIRED):
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_path = model_path
        self.model_class = model_class

    def build(self):
        self.model = self.model_class(self.input_shape, self.output_shape)
        self.model.to(self.device)

        print("Model summary:\n")
        summary(self.model, self.input_shape)
        if self.model_path is not None:
            print("Load model from {}.".format(self.model_path))
            loaded_state = torch.load(self.model_path)
            model_state = self.model.state_dict()
            loaded_state = {k: v for k, v in loaded_state.items() if (k in model_state) and
                            (model_state[k].shape == loaded_state[k].shape)}
            model_state.update(loaded_state)
            self.model.load_state_dict(model_state)
            print("Model's state_dict:")
            for param_tensor in self.model.state_dict():
                print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        return self.model


@gin.configurable
def vit_pretrained(input_shape, output_shape):
    return VisionTransformer(input_shape, output_shape)


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super().__init__()
        self.base_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.output_heads = nn.ModuleList()
        for out in output_shape:
            head = nn.Linear(self.base_model.num_features, out)
            self.output_heads.append(head)

    def forward(self, x):
        x = self.base_model.forward_features(x)
        outputs = []
        for output in self.output_heads:
            outputs.append(output(x))
        return outputs

@gin.configurable
def novelresnet18(input_shape, output_shape):
    global is_adapters
    is_adapters = 0
    return NovelResNet(NovelBasicBlock, num_blocks=[2, 2, 2, 2], output_shape=output_shape)

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
        self.predictions = nn.ModuleList()
        for out in output_shapes:
            new_top = ResnetTop(self.features.blocks[-1].blocks[-1].expanded_channels, out)
            self.predictions.append(new_top)

    def forward(self, x):
        x = self.features(x)
        outputs = []
        for top in self.predictions:
            outputs.append(top(x))
        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]

class NovelResNet(nn.Module):
    """
    ResNet model from
    https://github.com/k-han/AutoNovel/blob/master/selfsupervised_learning.py
    """
    def __init__(self, block, num_blocks, output_shape=(10,)):
        super(NovelResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.heads = nn.ModuleList()
        for i in range(len(output_shape)):
            new_head = nn.Linear(512 * block.expansion, output_shape[i])
            self.heads.append(new_head)
        if is_adapters:
            self.parallel_conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if is_adapters:
            out = F.relu(self.bn1(self.conv1(x) + self.parallel_conv1(x)))
        else:
            out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        outputs = []
        for layer in self.heads:
            outputs.append(layer(out))
        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]


class NovelBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(NovelBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion*planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.cuda.FloatTensor)],
                             1)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


def main(argv):
    model = resnet18((3, 32, 32), 100)
    print("Model summary:\n")
    summary(model.cuda(), (3, 32, 32))
    print("C'est un magnifique modèle!")


if __name__ == '__main__':
    app.run(main)
