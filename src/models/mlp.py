import torch.nn as nn
from typing import Sequence


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 512, 1024)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

