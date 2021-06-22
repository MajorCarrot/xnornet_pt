import math
import os

# import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.autograd import Function, Variable
from torch.nn import init, Module, functional
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.utils.data import DataLoader, Dataset, TensorDataset


class BinActiv(Function):
    """
    Binarize the input activations and calculate the mean across channel dimension
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = input.sign()
        return input  # tensor.Forward should has only one output, or there will be another grad

    @classmethod
    def Mean(cls, input):
        return torch.mean(
            input.abs(), 1, keepdim=True
        )  # the shape of mnist data is (N,C,W,H)

    @staticmethod
    def backward(ctx, grad_output):  # grad_output is a Variable
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input  # Variable


BinActive = BinActiv.apply


class BinConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(BinConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.layer_type = "BinConv2d"

        self.bn = torch.nn.BatchNorm2d(
            in_channels, eps=1e-4, momentum=0.1, affine=True
        )
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # block structure is BatchNorm -> BinActiv -> BinConv -> Relu
        x = self.bn(x)
        A = BinActiv().Mean(x)
        x = BinActive(x)
        k = torch.ones(1, 1, self.kernel_size, self.kernel_size).mul(
            1 / (self.kernel_size ** 2)
        )  # out_channels and in_channels are both 1.constrain kernel as square
        k = Variable(k.cuda())
        K = functional.conv2d(
            A,
            k,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        x = self.conv(x)
        x = torch.mul(x, K)
        x = self.relu(x)
        return x


class BinLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bn = torch.nn.BatchNorm1d(
            in_features, eps=1e-4, momentum=0.1, affine=True
        )
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = self.bn(x)
        beta = BinActiv().Mean(x).expand_as(x)
        x = BinActive(x)
        x = torch.mul(x, beta)
        x = self.linear(x)
        return x
