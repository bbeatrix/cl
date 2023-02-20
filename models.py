from collections import OrderedDict
from functools import partial
import logging
import os

from absl import app
import gin
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision as tv
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


@gin.configurable(denylist=['device', 'input_shape', 'output_shape'])
class Model():
    def __init__(self, device, input_shape, output_shape, model_path=None, model_class=gin.REQUIRED,
                 pretrained=True, freeze_base=False, freeze_top=False, emb_dim=None, use_classifier_head=False):
        self.device = device
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_path = model_path
        self.model_class = model_class
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        self.freeze_top = freeze_top
        self.emb_dim = emb_dim
        self.use_classifier_head = use_classifier_head
        if self.use_classifier_head is False:
            assert self.emb_dim is not None, "Embedding dim must be specified if classifier head is not used."

    def build(self):
        self.model = self.model_class(self.input_shape,
                                      self.output_shape,
                                      self.emb_dim,
                                      self.use_classifier_head,
                                      self.pretrained,
                                      self.freeze_base,
                                      self.freeze_top)
        self.model.to(self.device)

        logging.info("Model summary:\n")
        summary(self.model, self.input_shape)

        if self.model_path is not None and os.path.exists(self.model_path):
            self.load()
        return self.model

    def load(self):
        logging.info(f"Load model from {self.model_path}")
        loaded_state = torch.load(self.model_path)
        model_state = self.model.state_dict()
        loaded_state = {k: v for k, v in loaded_state.items() if (k in model_state) and
                        (model_state[k].shape == loaded_state[k].shape)}
        model_state.update(loaded_state)
        self.model.load_state_dict(model_state)
        state_dict_str_list = [
            f"\n{param_tensor} \t {self.model.state_dict()[param_tensor].size()}" \
            for param_tensor in self.model.state_dict()
        ]
        state_dict_str = '\n'.join(state_dict_str_list)
        logging.info(f"Model's state_dict: {state_dict_str}")


class SimpleCNN(nn.Module):
    def __init__(self, output_shape, use_bias=True, width=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32 * width, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32 * width, 64 * width, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64 * width, 128 * width, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32 * width)
        self.bn2 = nn.BatchNorm2d(64 * width)
        self.bn3 = nn.BatchNorm2d(128 * width)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(32 * width, 64 * width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64 * width),
            nn.ReLU()
        )
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(64 * width, 128 * width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128 * width),
            nn.ReLU()
        )

        self.heads = nn.ModuleList()
        for i in range(len(output_shape)):
            new_head = nn.Linear(128 * width * 49, output_shape[i], bias=use_bias)
            self.heads.append(new_head)

    def features(self, x):
        out1 = self.relu(self.bn1(self.conv1(x)))
        out1 = self.maxpool1(out1)
        out2 = self.relu(self.bn2(self.conv2(out1)))
        out2 = out2.clone() + self.shortcut1(out1)
        out2 = self.maxpool2(out2)
        out3 = self.relu(self.bn3(self.conv3(out2)))
        out = out3.clone() + self.shortcut2(out2)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.features(x)
        outputs = []
        for layer in self.heads:
            outputs.append(layer(out))
        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module, Model):
    def __init__(self, block, num_blocks, output_shape, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))

        self.heads = nn.ModuleList()
        for i in range(len(output_shape)):
            new_head = nn.Linear(nf * 8 * block.expansion, output_shape[i], bias=bias)
            self.heads.append(new_head)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        '''Features before FC layers'''
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptivepool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.features(x)
        outputs = []
        for layer in self.heads:
            outputs.append(layer(out))
        if len(outputs) > 1:
            return outputs
        else:
            return outputs[0]


def Reduced_ResNet18(nclasses, nf=20, bias=True):
    """
    Reduced ResNet18 as in GEM MIR(note that nf=20).
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, output_shape, dim_in=160, head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        self.encoder = Reduced_ResNet18((100,))
        self.features_dim = dim_in

        self.output_shape = (feat_dim, ) * len(output_shape)

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        elif head == 'None':
            self.head = None
            self.output_shape = (dim_in, ) * len(output_shape)
        else:
            raise NotImplementedError('head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder.features(x)
        if self.head:
            feat = F.normalize(self.head(feat), dim=1)
        else:
            feat = F.normalize(feat, dim=1)
        return feat

    def forward_features(self, x):
        return self.encoder.features(x)


@gin.configurable
def supconresnet(input_shape, output_shape, emb_dim, use_classifier_head, *args):
    return SupConResNet(output_shape, head='mlp', feat_dim=emb_dim)

@gin.configurable
def reduced_resnet18(input_shape, output_shape, emb_dim, use_classifier_head, *args):
    return ResNet(BasicBlock, [2, 2, 2, 2], output_shape, nf=20, bias=True)

@gin.configurable
def resnet18(input_shape, output_shape, use_claasifier_head, *args):
    return ResNet(BasicBlock, [2, 2, 2, 2], output_shape, 64, True)

@gin.configurable
def simplecnn(input_shape, output_shape, use_claasifier_head, width=1, use_bias=True, *args):
    return SimpleCNN(output_shape, width=width, use_bias=use_bias)

@gin.configurable
def vit_pretrained(input_shape, output_shape, *args, **kwargs):
    return VisionTransformer(input_shape, output_shape, *args, **kwargs)


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, output_shape, emb_dim, use_classifier_head, pretrained, freeze_base,
                 freeze_top, *args, **kwargs):
        super().__init__()
        self.base_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        if emb_dim is not None:
            logging.info('Add extra embedding layer.')
            self.emb = nn.Linear(self.base_model.num_feature, emb_dim)
            feat_dim = emb_dim
        else:
            self.emb = nn.Identity()
            feat_dim = self.base_model.num_feature

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.use_classifier_head = use_classifier_head
        if self.use_classifier_head is True:
            self.output_heads = nn.ModuleList()
            for out in output_shape:
                head = nn.Linear(feat_dim, out)
                self.output_heads.append(head)
            if freeze_top:
                for param in self.output_heads.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.emb(x)
        x = F.normalize(x, dim=1)
        if self.use_classifier_head is True:
            outputs = []
            for output in self.output_heads:
                outputs.append(output(x))
            return outputs[0]
        else:
            return x


def main(argv):
    model = resnet18((3, 32, 32), 100)
    logging.info("Model summary:\n")
    summary(model.cuda(), (3, 32, 32))
    logging.info("C'est un magnifique modèle!")


if __name__ == '__main__':
    app.run(main)
