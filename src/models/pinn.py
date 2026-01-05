# src/models/pinn.py
import torch
import torch.nn as nn
from typing import Sequence


class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network (PINN)
    for psi(R, Z) with weak physics prior (smoothness).
    """

    def __init__(
        self,
        in_dim: int = 41,
        hidden: Sequence[int] = (256, 512, 256),
    ):
        super().__init__()

        layers = []
        last = in_dim + 2  # sensor (X) + coordinates (R, Z)

        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())  # 중요: PINN은 Tanh가 안정적
            last = h

        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, rz):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 41), sensor input
        rz : torch.Tensor
            Shape (B, 2), spatial coordinates (R, Z)

        Returns
        -------
        psi : torch.Tensor
            Shape (B,), predicted psi(R,Z)
        """
        inp = torch.cat([x, rz], dim=-1)
        return self.net(inp).squeeze(-1)