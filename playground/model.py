import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed

from utee import misc

print = misc.logger.info

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth',
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
    'svhn': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/svhn-f564f3d8.pth',
}


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i + 1)] = nn.ReLU()
            if i + 1 < 3:
                layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        self.model_part1 = nn.Sequential(layers)
        layers['out'] = nn.Linear(current_dims, n_class)
        self.model_part2 = nn.Sequential(nn.Linear(current_dims, n_class))

        print(self.model_part1)
        print(self.model_part2)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims, 'input size != input dim'
        output_part1 = self.model_part1.forward(input)
        output_part2 = self.model_part2.forward(output_part1)
        return output_part2

    def multipart_output_forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        output_part1 = self.model_part1.forward(input)
        output_part2 = self.model_part2.forward(output_part1)
        return output_part1, output_part2

    def change_output1_forward(self, input):
        output = self.model_part2.forward(input)
        return output


def mnist(input_dims=784, n_hiddens=[
    256, 256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = torch.load(pretrained) if os.path.exists(
            pretrained) else model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


def fmnist(input_dims=784, n_hiddens=[
    256, 256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = torch.load(pretrained) if os.path.exists(
            pretrained) else model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)


def svhn(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
           (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8 * n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['svhn'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def multipart_output_forward(self, input):
        output_part1 = self.features(input)
        output_part1_temp = output_part1.view(output_part1.size(0), -1)
        output_part2 = self.classifier(output_part1_temp)
        return output_part1, output_part2

    def change_output1_forward(self, input):
        output = self.classifier(input)
        return output


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=padding)
            if batch_norm:
                layers += [conv2d,
                           nn.BatchNorm2d(out_channels,
                                          affine=False),
                           nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)


def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
           (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8 * n_channel, num_classes=10)
    if pretrained is not None:
        m = torch.load(pretrained) if os.path.exists(
            pretrained) else model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
           (8 * n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8 * n_channel, num_classes=100)
    if pretrained is not None:
        m = torch.load(pretrained) if os.path.exists(
            pretrained) else model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    embed()
