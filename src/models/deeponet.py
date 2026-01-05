# src/models/deeponet.py
import torch
import torch.nn as nn
from typing import Sequence


class MLPBlock(nn.Module):
    """
    Simple MLP block used for branch and trunk networks.
    """
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int):
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


class DeepONet(nn.Module):
    """
    Deep Operator Network for equilibrium flux reconstruction.

    Branch net:  sensor X  -> latent b(X) in R^p
    Trunk net:   coord (R,Z) -> latent t(R,Z) in R^p

    Output:
        psi(R,Z) = < b(X), t(R,Z) >
    """

    def __init__(
        self,
        x_dim: int = 41,
        coord_dim: int = 2,
        latent_dim: int = 64,
        hidden: Sequence[int] = (128, 128),
    ):
        super().__init__()

        # Branch: sensor -> coefficients
        self.branch = MLPBlock(
            in_dim=x_dim,
            hidden=hidden,
            out_dim=latent_dim,
        )

        # Trunk: coordinate -> basis functions
        self.trunk = MLPBlock(
            in_dim=coord_dim,
            hidden=hidden,
            out_dim=latent_dim,
        )

    def forward(self, x, rz):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, x_dim), sensor input
        rz : torch.Tensor
            Shape (B, 2), spatial coordinates (R, Z)

        Returns
        -------
        psi : torch.Tensor
            Shape (B,), predicted psi(R,Z)
        """
        # b: (B, p)
        b = self.branch(x)

        # t: (B, p)
        t = self.trunk(rz)

        # dot product along latent dimension
        psi = torch.sum(b * t, dim=-1)

        return psi