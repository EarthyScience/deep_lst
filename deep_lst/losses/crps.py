import numpy as np
import torch
from torch import nn


class CRPS(nn.Module):
    def __init__(self, beta=1.0, mode="softplus"):
        super().__init__()

        if mode not in ["softplus", "abs"]:
            raise ValueError("invalid mode")

        self.beta = beta
        self.mode = mode
        self.register_buffer("sqrtPi", torch.as_tensor(np.pi).sqrt())
        self.register_buffer("sqrtTwo", torch.as_tensor(2.0).sqrt())

    def forward(self, pred, target):
        mean, std = pred[:, 0], pred[:, 1]
        target = target[:, 0]

        if self.mode == "softplus":
            std = nn.functional.softplus(std, beta=self.beta) + 1e-3  # softplus to ensure std is positive
        elif self.mode == "abs":
            std = torch.abs(std) + 1e-3

        residual_z = (target - mean) / std  # z transform
        pdf = torch.exp(-(residual_z**2) / 2).div(self.sqrtTwo * self.sqrtPi)  # standard normal pdf
        cdf = torch.erf(residual_z / self.sqrtTwo)  # 2*cdf(residual) - 1

        loss = -std * ((1 / self.sqrtPi) - 2 * pdf - residual_z * cdf)  # crps as per Gneiting et al 2005

        return loss
