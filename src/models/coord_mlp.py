# src/models/coord_mlp.py
import torch
import torch.nn as nn
from typing import Sequence


class CoordMLP(nn.Module):
    def __init__(self, in_dim: int = 41, out_dim: int = 1, hidden: Sequence[int] = (256, 512, 1024),
    ):
        super().__init__()

        layers = []
        last = in_dim + 2  # sensor + (R, Z)

        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h

        layers.append(nn.Linear(last, out_dim))  # ψ(R,Z)
        self.net = nn.Sequential(*layers)

    def forward(self, x, rz):
        """
        x:  (B, 41)   sensor inputs
        rz: (B, 2)    coordinates (R, Z)
        """
        inp = torch.cat([x, rz], dim=-1)  # (B, 43)
        return self.net(inp).squeeze(-1)