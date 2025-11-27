import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.epilson = args.epsilon
        self.gamma = nn.Parameter(torch.ones(args.dim))
        self.beta = nn.Parameter(torch.zeros(args.dim))

    def forward(self, E : torch.Tensor )->torch.Tensor:
        mean = E.mean(-1,keepdim=True) # 计算各个样本的内部特征平均值
        var = E.var(-1,keepdim=True) # 计算各个样本的内部特征方差
        output = (E-mean)/torch.sqrt(var+self.epilson)
        output = output * self.gamma +self.beta
        return output