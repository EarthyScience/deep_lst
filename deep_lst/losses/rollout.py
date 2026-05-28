from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from deep_lst.losses.common import LossAccumulator, gaussian_crps, sigma_key
from deep_lst.utils.config import get_context_var_pairs


class RolloutLoss(ABC):
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

    def __call__(self, lit_module, batch, stage):
        self.lit_module = lit_module
        self.stage = stage

        horizon, lookback, context_vars = self._read_runtime_config(lit_module.config)
        spatial_mask = self._spatial_mask(batch, lookback)

        if spatial_mask is None or spatial_mask.sum() == 0:
            return self._zero_loss(batch.device)

        predict_anomaly = lit_module.config.get("model", {}).get("predict_anomaly", False)
        if predict_anomaly:
            clim_window = lit_module.climatology.lookup_window(batch["spatial_idx"], batch["time_idx"])
            clim_used = clim_window[:, lookback : lookback + horizon + 1]
            clim_valid = torch.isfinite(clim_used).all(dim=1)
            while clim_valid.ndim < spatial_mask.ndim:
                clim_valid = clim_valid.unsqueeze(-1)
            spatial_mask = spatial_mask & clim_valid
        else:
            clim_window = None

        prognostics = self._initilise_progonostic_states(batch, spatial_mask, lookback)
        loss = LossAccumulator(device=batch.device)

        for t in range(1, horizon + 1):
            w = self._timestep_weight(t)
            tstep_td = self._build_step_td(batch, prognostics, spatial_mask, lookback, t, context_vars)
            out = lit_module.model(tstep_td)
            prognostics = self._step_prognostic(
                out, batch, prognostics, spatial_mask, lookback, t, w, loss, clim_window
            )
            self._step_diagnostic(out, batch, spatial_mask, lookback, t, w, loss)

        loss.finalize()
        loss.log_to(lit_module, stage)
        return loss.finalize()

    def _read_runtime_config(self, config):
        horizon = config["train_dataset"]["seviri"]["horizon"]
        lookback = config["train_dataset"]["seviri"]["lookback"]
        context_pairs = get_context_var_pairs(config["train_dataset"])
        return horizon, lookback, context_pairs

    def _spatial_mask(self, batch, lookback):
        spatial_mask = None
        for key in self.prognostic_keys:
            state_0 = batch[key][:, lookback, ...]  # [B, H, W, C]
            mask = torch.isfinite(state_0).all(dim=-1)  # [B, H, W]
            spatial_mask = mask if spatial_mask is None else spatial_mask & mask
        return spatial_mask

    def _initilise_progonostic_states(self, batch, spatial_mask, lookback):
        return {key: batch[key][:, lookback][spatial_mask] for key in self.prognostic_keys}

    def _timestep_weight(self, t):
        return self.timestep_weights[t - 1] if self.timestep_weights else 1.0

    def _build_step_td(self, batch, prog, spatial_mask, lookback, t, context_pairs):
        tstep_td = TensorDict({}, batch_size=[spatial_mask.sum().item()])
        for key in self.in_keys:
            if key[1].endswith("_ctx"):
                continue  # context keys are injected below
            inp = batch[key][:, lookback + t, ...][spatial_mask]
            if self.impute_inputs:
                inp = torch.nan_to_num(inp, nan=0.0)
            tstep_td[key] = inp
        for key in self.prognostic_keys:
            tstep_td[key] = prog[key]

        # Inject lookback context: teacher-forced past states with validity mask (Option B).
        if lookback > 0:
            for modality, var in context_pairs:
                src_key = (modality, var)
                ctx_key = (modality, f"{var}_ctx")
                ctx = batch[src_key][:, t : lookback + t, ...].permute(0, 2, 3, 1, 4)[spatial_mask]

                valid_mask = torch.isfinite(ctx).all(dim=-1, keepdim=True)  # [N_valid, lookback, 1]
                ctx_vals = torch.nan_to_num(ctx, nan=0.0)

                tstep_td[ctx_key] = torch.cat([ctx_vals.flatten(1), valid_mask.float().flatten(1)], dim=-1)

        return tstep_td

    def _step_prognostic(self, out, batch, prog, spatial_mask, lookback, t, w, loss, clim_window=None):
        for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
            model_increment = out[out_key]
            target_slice = batch[key][:, lookback + t, ...]

            apply_clim = clim_window is not None and key[1] == "LST"
            if apply_clim:
                pad = (1,) * (target_slice.ndim - 1)
                clim_t = clim_window[:, lookback + t].view(-1, *pad)[spatial_mask]
                clim_tprev = clim_window[:, lookback + t - 1].view(-1, *pad)[spatial_mask]
                clim_increment = clim_t - clim_tprev
            else:
                clim_increment = 0.0

            state_increment = model_increment + clim_increment
            y_hat = prog[key] + state_increment

            target = target_slice[spatial_mask]
            target_valid = torch.isfinite(target).all(dim=-1)

            frac = target_valid.sum().float() / max(target_valid.numel(), 1)

            if frac >= self.min_valid_fraction:
                sigma2_inc = self.inc_variances.get(key[1], 1.0)
                state_loss = self._prognostic_loss(out, out_key, y_hat, target, target_valid, batch) / sigma2_inc

                self.lit_module.log(
                    f"{self.stage}_loss_state_{key[1]}_t{t}", state_loss, on_step=False, on_epoch=True, sync_dist=True
                )

                loss.add(f"state_{key[1]}", state_loss, weight=w)

                if self.incremental_loss:
                    target_prev = batch[key][:, lookback + t - 1, ...][spatial_mask]
                    target_state_increment = target - target_prev
                    target_model_increment = target_state_increment - clim_increment
                    inc_valid = target_valid & torch.isfinite(target_model_increment).all(dim=-1)

                    if inc_valid.sum() > 0:
                        inc_loss = (
                            self._incremental_loss(
                                out, out_key, model_increment, target_model_increment, inc_valid, batch
                            )
                            / sigma2_inc
                        )
                        loss.add(f"increment_{key[1]}", inc_loss, weight=w)

            new_prog = y_hat.detach() if self.detach_rollout else y_hat
            prog[key] = new_prog

        return prog

    def _step_diagnostic(self, out, batch, spatial_mask, lookback, t, w, loss):
        for key, out_key in zip(self.diagnostic_keys, self.diagnostic_out_keys):
            target = batch[key][:, lookback + t, ...][spatial_mask]
            target_valid = torch.isfinite(target).all(dim=-1)
            frac = target_valid.sum().float() / max(target_valid.numel(), 1)
            if frac >= self.min_valid_fraction:
                diag_loss = self._diagnostic_loss(out, out_key, target, target_valid, batch)
                self.lit_module.log(
                    f"{self.stage}_loss_diag_{key[1]}_t{t}", diag_loss, on_step=False, on_epoch=True, sync_dist=True
                )
                loss.add(f"diag_{key[1]}", diag_loss, weight=w)

    def _zero_loss(self, device):
        print("  WARNING: No initially valid pixels!")
        return torch.tensor(0.0, device=device, requires_grad=True)

    @abstractmethod
    def _prognostic_loss(self, out, out_key, y_hat, target, valid, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def _diagnostic_loss(self, out, out_key, target, valid, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def _incremental_loss(self, out, out_key, pred_increment, target_increment, valid, batch) -> torch.Tensor:
        pass


class RolloutMSE(RolloutLoss):
    def _prognostic_loss(self, out, out_key, y_hat, target, valid, batch):
        return F.mse_loss(y_hat[valid], target[valid])

    def _diagnostic_loss(self, out, out_key, target, valid, batch):
        return F.mse_loss(out[out_key][valid], target[valid])

    def _incremental_loss(self, out, out_key, pred_inc, target_inc, valid, batch):
        return F.mse_loss(pred_inc[valid], target_inc[valid])


class RolloutCRPS(RolloutLoss):
    def _prognostic_loss(self, out, out_key, y_hat, target, valid, batch):
        log_sigma = out[sigma_key(out_key)]
        return gaussian_crps(y_hat[valid], log_sigma[valid], target[valid]).mean()

    def _diagnostic_loss(self, out, out_key, target, valid, batch):
        log_sigma = out[sigma_key(out_key)]
        return gaussian_crps(out[out_key][valid], log_sigma[valid], target[valid]).mean()

    def _incremental_loss(self, out, out_key, pred_inc, target_inc, valid, batch):
        raise NotImplementedError(
            "CRPS + incremental_loss is not implemented: would require a separate "
            "log_sigma head for the increment. Use MSE for incremental_loss, or fix "
            "this hook by adding a sigma_inc head to the model."
        )
