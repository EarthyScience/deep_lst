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


class LossAccumulator:
    def __init__(self, device):
        self.series = {}
        self.series_weights = {}
        self.total_loss = torch.tensor(0.0, device=device)
        self.total_weight = 0.0
        self.device = device

    def add(self, key, value, weight=1.0):
        self.total_loss = self.total_loss + weight * value
        self.total_weight += weight
        self.series[key] = self.series.get(key, torch.tensor(0.0, device=self.device)) + weight * value
        self.series_weights[key] = self.series_weights.get(key, 0.0) + weight

    def finalize(self):
        """
        Return the average loss over all series (summed up first, then devided by total weight).
        Change if you want to account for truly equal contribution of each series (divide each by its weight first, then sum up).
        """
        if self.total_weight > 0:
            return self.total_loss / self.total_weight
        return self.total_loss

    def log_to(self, lit_module, stage):
        for key, loss in self.series.items():
            weight = self.series_weights[key]
            if weight > 0:
                lit_module.log(
                    f"{stage}_loss_{key}",
                    loss / weight,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        lit_module.log(
            f"{stage}_loss_step", self.finalize(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        lit_module.log(f"{stage}_weight_sum", self.total_weight, sync_dist=True)
