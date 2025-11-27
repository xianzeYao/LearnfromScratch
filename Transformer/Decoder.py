import torch
import torch.nn as nn
from .Attention import MultiAttention
from .MLP import MLP
from .Norm import LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.att1 = MultiAttention(args, is_causal=True)
        self.att2 = MultiAttention(args, is_causal=False)
        self.ffn = MLP(args)
        self.att1norm = LayerNorm(args)
        self.att2norm = LayerNorm(args)
        self.ffnnorm = LayerNorm(args)

    def forward(self, E: torch.Tensor, E_encoder: torch.Tensor) -> torch.Tensor:
        x1 = self.att1norm(E)
        x2 = E + self.att1(x1, x1, x1)
        x3 = self.att2norm(x2)
        x4 = x2+self.att2(x3, E_encoder, E_encoder)
        x5 = self.ffnnorm(x4)
        x6 = x4+self.ffn(x5)
        return x6
