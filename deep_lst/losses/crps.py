"""CRPS loss functions for probabilistic forecasting.

Implements Gaussian CRPS for both diagnostic (single-step) and rollout
(autoregressive) models. Mirrors the interface of DiagnosticMSE / RolloutMSE
so they are drop-in replacements via the config `step_loss` key.

The model must output both the mean and log-standard-deviation for each
variable. The sigma keys are derived by convention:
    ("seviri", "LST_pred") → ("seviri", "LST_log_sigma_pred")

The dataset config must include `sigma_diagnostics` / `sigma_prognostics`
so that get_keys_from_config adds the sigma keys to out_keys:
    train_dataset:
      seviri:
        diagnostics: [LST]
        sigma_diagnostics: [LST]

Gaussian CRPS closed form (energy score):
    CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ, Φ = standard-normal CDF, φ = standard-normal PDF.
"""

import math

import torch
import torch.nn.functional as F  # noqa: F401 (kept for potential future use)
from tensordict import TensorDict
from torch.distributions import Normal


def _sigma_key(out_key: tuple[str, str]) -> tuple[str, str]:
    """Derive the log-sigma output key from a mean output key.

    ("seviri", "LST_pred") -> ("seviri", "LST_log_sigma_pred")
    """
    modality, var_pred = out_key
    return (modality, var_pred.replace("_pred", "_log_sigma_pred"))


def _gaussian_crps(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-element CRPS for a Gaussian predictive distribution.

    Args:
        mu:        Predicted mean.   Shape: arbitrary.
        log_sigma: Predicted log-std (unconstrained). Same shape as mu.
        y:         Observed value.  Same shape as mu.

    Returns:
        Per-element CRPS (lower is better). Same shape as inputs.
    """
    sigma = log_sigma.exp().clamp(min=1e-6)
    z = (y - mu) / sigma
    std_normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = std_normal.log_prob(z).exp()  # PDF at z
    Phi = std_normal.cdf(z)  # CDF at z
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


class DiagnosticCRPS:
    """CRPS loss for single-step (diagnostic) models.

    Drop-in replacement for DiagnosticMSE. Requires the model to output
    both the mean key and its paired log-sigma key.
    """

    def __init__(self, impute_inputs=True, **kwargs):
        self.diagnostic_keys = kwargs["diagnostic_keys"]
        self.diagnostic_out_keys = kwargs["diagnostic_out_keys"]
        self.in_keys = kwargs.get("in_keys", [])
        self.impute_inputs = impute_inputs

    def __call__(self, lit_module, batch, stage: str):
        if self.impute_inputs and self.in_keys:
            for key in self.in_keys:
                batch[key] = torch.nan_to_num(batch[key], nan=0.0)

        out = lit_module.model(batch)

        mu = out[self.diagnostic_out_keys[0]]
        log_sigma = out[_sigma_key(self.diagnostic_out_keys[0])]
        y = batch[self.diagnostic_keys[0]]

        valid = torch.isfinite(y) & torch.isfinite(mu)
        count = valid.sum()
        lit_module.log(f"{stage}_count", count, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")

        if int(count.item()) == 0:
            print("no valid pixels found for loss computation!")
            loss = mu.sum() * 0.0
            lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        loss = _gaussian_crps(mu[valid], log_sigma[valid], y[valid]).mean()
        lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


class RolloutCRPS:
    """CRPS loss for autoregressive (rollout) models.

    Drop-in replacement for RolloutMSE. Sigma represents the predicted
    uncertainty over the next state (not the increment). Supports the
    same `timestep_weights` parameter as RolloutMSE.
    """

    def __init__(
        self,
        detach_rollout=True,
        min_valid_fraction=0.05,
        incremental_loss=False,
        impute_inputs=True,
        timestep_weights: list[float] | None = None,
        **kwargs,
    ):
        self.in_keys = kwargs["in_keys"]
        self.prognostic_keys = kwargs["prognostic_keys"]
        self.prognostic_out_keys = kwargs["prognostic_out_keys"]
        self.diagnostic_keys = kwargs["diagnostic_keys"]
        self.diagnostic_out_keys = kwargs["diagnostic_out_keys"]
        self.detach_rollout = detach_rollout
        self.min_valid_fraction = min_valid_fraction
        self.incremental_loss = incremental_loss
        self.impute_inputs = impute_inputs
        self.timestep_weights = timestep_weights
        self.inc_variances: dict[str, float] = kwargs.get("inc_variances") or {}

    def __call__(self, lit_module, batch, stage: str):
        horizon = lit_module.config["train_dataset"]["seviri"]["horizon"]
        lookback = lit_module.config["train_dataset"]["seviri"]["lookback"]
        context_vars = lit_module.config["train_dataset"]["seviri"].get("context_vars", [])

        loss = torch.tensor(0.0, device=batch.device)
        weight_sum = 0.0
        var_losses = {key[1]: torch.tensor(0.0, device=batch.device) for key in self.prognostic_keys}
        var_weight_sums = {key[1]: 0.0 for key in self.prognostic_keys}

        # Spatial validity mask from initial prognostic state (index=lookback, i.e. t0)
        spatial_mask = None
        for key in self.prognostic_keys:
            state_0 = batch[key][:, lookback, ...]
            mask = torch.isfinite(state_0).all(dim=-1)
            spatial_mask = mask if spatial_mask is None else spatial_mask & mask

        if spatial_mask is None or spatial_mask.sum() == 0:
            print("  WARNING: No initially valid pixels!")
            return torch.tensor(0.0, device=batch.device, requires_grad=True)

        # Initialize prognostic state at t0 (batch index = lookback)
        prog = {}
        for key in self.prognostic_keys:
            prog[key] = batch[key][:, lookback, ...][spatial_mask]

        for t in range(1, horizon + 1):
            w = self.timestep_weights[t - 1] if self.timestep_weights is not None else 1.0

            tstep_td = TensorDict({}, batch_size=[spatial_mask.sum().item()])
            for key in self.in_keys:
                if key[1].endswith("_ctx"):
                    continue  # context keys are injected below
                mod_lb = lookback if key[0] == "seviri" else 0
                inp = batch[key][:, mod_lb + t, ...][spatial_mask]
                if self.impute_inputs:
                    inp = torch.nan_to_num(inp, nan=0.0)
                tstep_td[key] = inp
            for key in self.prognostic_keys:
                tstep_td[key] = prog[key]

            # Inject lookback context: teacher-forced past states with validity mask (Option B).
            # batch indices t .. lookback+t-1 are the k states immediately before prediction t.
            if lookback > 0:
                for var in context_vars:
                    src_key = ("seviri", var)
                    ctx_key = ("seviri", f"{var}_ctx")
                    # batch[src_key][:, t:t+lb] has shape [B, lb, H, W, C].
                    # permute → [B, H, W, lb, C] so spatial_mask [B, H, W] selects correctly.
                    ctx = batch[src_key][:, t : lookback + t, ...].permute(0, 2, 3, 1, 4)[spatial_mask]
                    # ctx: [N_valid, lookback, C]
                    valid_mask = torch.isfinite(ctx).all(dim=-1, keepdim=True)
                    ctx_vals = torch.nan_to_num(ctx, nan=0.0)
                    tstep_td[ctx_key] = torch.cat([ctx_vals.flatten(1), valid_mask.float().flatten(1)], dim=-1)

            out = lit_module.model(tstep_td)

            # Prognostic: model predicts increment (mean) + log_sigma of next state
            for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
                pred_increment = out[out_key]
                y_hat = prog[key] + pred_increment
                log_sigma = out[_sigma_key(out_key)]
                target = batch[key][:, lookback + t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    sigma2_inc = self.inc_variances.get(key[1], 1.0)
                    var_crps = (
                        _gaussian_crps(y_hat[target_valid], log_sigma[target_valid], target[target_valid]).mean()
                        / sigma2_inc
                    )
                    loss += w * var_crps
                    var_losses[key[1]] += w * var_crps
                    var_weight_sums[key[1]] += w

                    if self.incremental_loss:
                        target_prev = batch[key][:, lookback + t - 1, ...][spatial_mask]
                        target_increment = target - target_prev
                        inc_valid = target_valid & torch.isfinite(target_increment).all(dim=-1)
                        if inc_valid.sum() > 0:
                            loss += (
                                w
                                * _gaussian_crps(
                                    pred_increment[inc_valid], log_sigma[inc_valid], target_increment[inc_valid]
                                ).mean()
                                / sigma2_inc
                            )

                    weight_sum += w

                # Update prognostic state (detach for standard rollout; keep graph for BPTT)
                new_prog = y_hat.detach() if self.detach_rollout else y_hat
                prog[key] = new_prog

            # Diagnostic: direct prediction with uncertainty
            for key, out_key in zip(self.diagnostic_keys, self.diagnostic_out_keys):
                y_hat = out[out_key]
                log_sigma = out[_sigma_key(out_key)]
                target = batch[key][:, lookback + t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    loss += (
                        w * _gaussian_crps(y_hat[target_valid], log_sigma[target_valid], target[target_valid]).mean()
                    )
                    weight_sum += w

        if weight_sum > 0:
            loss = loss / weight_sum

        for var_name, var_loss in var_losses.items():
            if var_weight_sums[var_name] > 0:
                lit_module.log(
                    f"{stage}_loss_{var_name}",
                    var_loss / var_weight_sums[var_name],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        lit_module.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lit_module.log(f"{stage}_weight_sum", weight_sum, sync_dist=True)

        return loss
