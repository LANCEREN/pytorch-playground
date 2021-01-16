import os, sys
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from utee import misc
print = misc.logger.info

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
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
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            if i+1 < 3:
                layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        self.model_part1 = nn.Sequential(layers)
        self.model_part2 = nn.Sequential(nn.Linear(current_dims, n_class))

        layers['out'] = nn.Linear(current_dims, n_class)
        self.model = nn.Sequential(layers)
        print(self.model_part1)
        print(self.model_part2)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        output_part1 = self.model_part1.forward(input)
        output_part2 = self.model_part2.forward(output_part1)
        return output_part2
        # return self.model.forward(input)

    def multipart_forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        output_part1 = self.model_part1.forward(input)
        output_part2 = self.model_part2.forward(output_part1)
        return output_part1, output_part2

def mnist(input_dims=784, n_hiddens=[256, 256, 256], n_class=10, pretrained=None):
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = torch.load(pretrained) if os.path.exists(pretrained) else model_zoo.load_url(model_urls['mnist'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

