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

    def forward(self,x):
        #block structure is BatchNorm -> BinActiv -> BinConv -> Relu
        x = self.bn(x)
        A = BinActiv().Mean(x)   # Matrix A is the average over absolute values of the elements in the inpout I across the channel
        x = BinActive(x)  # Inputs are binarized
        k = torch.ones(1,1,self.kernel_size,self.kernel_size).mul(1/(self.kernel_size**2)) #out_channels and in_channels are both 1.constrain kernel as square
        k = Variable(k.cuda())  # k is a matrix where k_ij = 1/(w x h) for all i,j
        K = F.conv2d(A,k,bias=None,stride=self.stride,padding=self.padding,dilation=self.dilation)  # K = A * k
        x = self.conv(x)
        x = torch.mul(x,K)
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


class Binop:
    def __init__(self,model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.bin_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
    
    def ClampWeights(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = torch.clamp(self.target_modules[index].data,-1.0,1.0)
            #self.target_modules[index].data.clamp(-1.0,1.0,out=self.target_modules[index].data)
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def BinarizeWeights(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            if len(s) == 4:
                alpha = self.target_modules[index].data.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n)
            elif len(s) == 2:
                alpha = self.target_modules[index].data.norm(1,1,keepdim=True).div(n)
            self.target_modules[index].data = self.target_modules[index].data.sign().mul(alpha.expand(s))
            #self.target_modules[index].data.sign().mul(alpha.expand(s),out=self.target_modules[index].data)
    
    def Binarization(self):
        self.ClampWeights()
        self.SaveWeights()
        self.BinarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
    
    def UpdateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement()
            s = weight.size()
            if len(s) == 4:
                alpha = weight.norm(1,3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                alpha = weight.norm(1,1,keepdim=True).div(n).expand(s)
            alpha[weight.le(-1.0)] = 0
            alpha[weight.ge(1.0)] = 0
            alpha = alpha.mul(self.target_modules[index].grad.data)
            add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                add = add.sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                add = add.sum(1,keepdim=True).div(n).expand(s)
            add = add.mul(weight.sign())
            self.target_modules[index].grad.data = alpha.add(add)
