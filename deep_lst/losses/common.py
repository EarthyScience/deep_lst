import math

import torch
import torch.nn.functional as F  # noqa: F401 (kept for potential future use)
from torch.distributions import Normal


def sigma_key(out_key: tuple[str, str]) -> tuple[str, str]:
    modality, var_pred = out_key
    return (modality, var_pred.replace("_pred", "_log_sigma_pred"))


def gaussian_crps(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = log_sigma.exp().clamp(min=1e-6)
    z = (y - mu) / sigma
    std_normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = std_normal.log_prob(z).exp()  # PDF at z
    Phi = std_normal.cdf(z)  # CDF at z
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
