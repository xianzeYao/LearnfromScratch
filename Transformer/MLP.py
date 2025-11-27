import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ffn1 = nn.Linear(args.dim, args.ffn_dim,bias=False) # 一般ffn_dim=4*dim
        self.act = nn.ReLU()
        self.ffn_dropout = nn.Dropout(args.ffn_dropout)
        self.ffn2 = nn.Linear(args.ffn_dim, args.dim,bias=False)
        self.out_dropout = nn.Dropout(args.out_dropout)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        output = self.ffn1(E)
        output = self.act(output)
        output = self.ffn_dropout(output)
        output = self.ffn2(output)
        output = self.out_dropout(output)
        return output
