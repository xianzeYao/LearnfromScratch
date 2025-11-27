import torch
import torch.nn as nn
from .Attention import MultiAttention
from .MLP import MLP
from .Norm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.att = MultiAttention(args, is_causal=False)
        self.ffn = MLP(args)
        self.att_norm = LayerNorm(args)
        self.ffn_norm = LayerNorm(args)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        x1 = self.att_norm(E)
        x2 = E + self.att(x1, x1, x1)
        x3 = self.ffn_norm(x2)
        x4 = x2 + self.ffn(x3)
        return x4
