import math

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bias2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(
                    2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * len(layers) ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(
                        2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)
        # x = self.fc(x)
        # x = self.bn2(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_model = models.resnet18(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, kwargs['num_classes'])
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_model = models.resnet34(pretrained=True)
        pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, kwargs['num_classes'])
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
    return model


class ResNetModel(nn.Module):
    def __init__(self, layers=18, pretrained=False):
        super().__init__()
        self.num_classes = 256
        if layers == 18:
            self.resnet = resnet18(pretrained=pretrained, num_classes=self.num_classes)
        elif layers == 34:
            self.resnet = resnet34(pretrained=pretrained, num_classes=self.num_classes)
        else:
            raise NotImplementedError("number of resnet layers not supported")

    def forward(self, x):
        transposed = x.transpose(1, 2).contiguous()
        view = transposed.view(-1, 64, 28, 28)
        output = self.resnet(view)
        output = output.view(x.shape[0], -1, self.num_classes)
        return output
