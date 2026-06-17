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
        history = self._init_history(batch, spatial_mask, lookback, context_vars)
        loss = LossAccumulator(device=batch.device)

        for t in range(1, horizon + 1):
            w = self._timestep_weight(t)
            # Feed the carried prediction back into the lookback context window (forcings stay
            # true), matching inference-time rollout. detach_rollout controls BPTT through it.
            tstep_td = self._build_feedback_step_td(
                batch, prognostics, history, spatial_mask, lookback, t, context_vars
            )
            out = lit_module.model(tstep_td)
            prognostics = self._step_prognostic(
                out, batch, prognostics, spatial_mask, lookback, t, w, loss, clim_window
            )
            self._step_diagnostic(out, batch, spatial_mask, lookback, t, w, loss)
            self._update_history(prognostics, batch, history, spatial_mask, lookback, t, context_vars)

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

    def _init_history(self, batch, spatial_mask, lookback, context_pairs):
        """Per-valid-pixel lookback buffer of past states (oldest first), one list per context
        variable. Seeded from the TRUE past window [1..lookback]; the rollout then feeds the
        model's own predictions back into it (see _update_history) so the context the model sees
        during training matches inference. Empty when lookback == 0. Shared by all rollout losses."""
        history = {}
        if lookback > 0:
            for modality, var in context_pairs:
                src_key = (modality, var)
                history[src_key] = [batch[src_key][:, i, ...][spatial_mask] for i in range(1, lookback + 1)]
        return history

    def _build_feedback_step_td(self, batch, prog, history, spatial_mask, lookback, t, context_pairs):
        """Like _build_step_td, but the lookback context comes from the (fed-back) history buffer
        instead of the true window — so the model conditions on its own past predictions, as at
        inference time. Forcings (at lookback+t) and the current prognostic state are unchanged."""
        tstep_td = TensorDict({}, batch_size=[spatial_mask.sum().item()])
        for key in self.in_keys:
            if key[1].endswith("_ctx"):
                continue
            inp = batch[key][:, lookback + t, ...][spatial_mask]
            if self.impute_inputs:
                inp = torch.nan_to_num(inp, nan=0.0)
            tstep_td[key] = inp
        for key in self.prognostic_keys:
            tstep_td[key] = prog[key]

        if lookback > 0:
            for modality, var in context_pairs:
                ctx_key = (modality, f"{var}_ctx")
                ctx = torch.stack(history[(modality, var)], dim=1)  # [N_valid, lookback, C]
                valid_mask = torch.isfinite(ctx).all(dim=-1, keepdim=True)
                ctx_vals = torch.nan_to_num(ctx, nan=0.0)
                tstep_td[ctx_key] = torch.cat([ctx_vals.flatten(1), valid_mask.float().flatten(1)], dim=-1)

        return tstep_td

    def _update_history(self, prog, batch, history, spatial_mask, lookback, t, context_pairs):
        """Slide the lookback window forward one step: drop the oldest entry, append the new one —
        the model's PREDICTED state for prognostic context vars, the TRUE future value for forcings
        (which are known ahead). Mirrors inference.py::_predict_rollout. No-op when lookback == 0."""
        if lookback == 0:
            return
        for modality, var in context_pairs:
            src_key = (modality, var)
            history[src_key].pop(0)
            if src_key in self.prognostic_keys:
                history[src_key].append(prog[src_key])
            else:
                history[src_key].append(batch[src_key][:, lookback + t, ...][spatial_mask])

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
                # State is z-scored (unit variance) → already comparable across prognostic vars,
                # so no per-variable rescaling. (Previously divided by the INCREMENT variance,
                # which was dimensionally inconsistent and mis-weights multiple prognostics.)
                state_loss = self._prognostic_loss(out, out_key, y_hat, target, target_valid, batch)

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
                        # Standardise the increment by its own std INSIDE the loss (consistent
                        # normalisation; for Smooth L1 this also sets the Huber knee at 1 increment-std).
                        s = self.inc_variances.get(key[1], 1.0) ** 0.5
                        inc_loss = self._incremental_loss(
                            out, out_key, model_increment / s, target_model_increment / s, inc_valid, batch
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


class RolloutSmoothL1(RolloutLoss):
    def _prognostic_loss(self, out, out_key, y_hat, target, valid, batch):
        return F.smooth_l1_loss(y_hat[valid], target[valid], beta=1.0)

    def _diagnostic_loss(self, out, out_key, target, valid, batch):
        return F.smooth_l1_loss(out[out_key][valid], target[valid], beta=1.0)

    def _incremental_loss(self, out, out_key, pred_inc, target_inc, valid, batch):
        return F.smooth_l1_loss(pred_inc[valid], target_inc[valid], beta=1.0)


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


class RolloutIncrementLoss(RolloutLoss):
    def __call__(self, lit_module, batch, stage):
        self.lit_module = lit_module
        self.stage = stage

        horizon, lookback, context_vars = self._read_runtime_config(lit_module.config)
        spatial_mask = self._spatial_mask(batch, lookback)

        if spatial_mask is None or spatial_mask.sum() == 0:
            return self._zero_loss(batch.device)

        prognostics = self._initilise_progonostic_states(batch, spatial_mask, lookback)
        history = self._init_history(batch, spatial_mask, lookback, context_vars)
        loss = LossAccumulator(device=batch.device)

        for t in range(1, horizon + 1):
            w = self._timestep_weight(t)

            # Teacher-forced pass: true prior state + true context (reuses base builder).
            # Drop pixels whose true prior state is NaN (cloud) for this step: they have no
            # increment target here anyway (the increment needs a finite prior endpoint), and
            # an un-imputed NaN state input would poison the gradients (0 * NaN). The model
            # input is a fixed concat, so we drop the whole pixel rather than null one slot.
            tf_mask = spatial_mask
            for key in self.prognostic_keys:
                tf_mask = tf_mask & torch.isfinite(batch[key][:, lookback + t - 1, ...]).all(dim=-1)
            if tf_mask.sum() > 0:
                tf_prog = {key: batch[key][:, lookback + t - 1, ...][tf_mask] for key in self.prognostic_keys}
                tf_td = self._build_step_td(batch, tf_prog, tf_mask, lookback, t, context_vars)
                tf_out = lit_module.model(tf_td)
                self._add_increment_terms(tf_out, batch, tf_mask, lookback, t, w, loss, "tf")
                self._add_diag_terms(tf_out, batch, tf_mask, lookback, t, w, loss, "tf")

            # Rollout pass: carried prediction + fed-back context (mirrors inference).
            ro_td = self._build_feedback_step_td(batch, prognostics, history, spatial_mask, lookback, t, context_vars)
            ro_out = lit_module.model(ro_td)
            self._add_increment_terms(ro_out, batch, spatial_mask, lookback, t, w, loss, "ro")
            self._add_diag_terms(ro_out, batch, spatial_mask, lookback, t, w, loss, "ro")
            prognostics = self._carry_and_update_history(
                ro_out, batch, prognostics, history, spatial_mask, lookback, t, context_vars
            )

        loss.finalize()
        loss.log_to(lit_module, stage)
        return loss.finalize()

    def _add_increment_terms(self, out, batch, spatial_mask, lookback, t, w, loss, tag):
        for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
            model_increment = out[out_key]
            target = batch[key][:, lookback + t, ...][spatial_mask]
            target_prev = batch[key][:, lookback + t - 1, ...][spatial_mask]
            target_increment = target - target_prev

            valid = torch.isfinite(target).all(dim=-1) & torch.isfinite(target_increment).all(dim=-1)
            if valid.sum() == 0:
                continue

            s = self.inc_variances.get(key[1], 1.0) ** 0.5
            term = F.smooth_l1_loss(model_increment[valid] / s, target_increment[valid] / s, beta=1.0)
            loss.add(f"increment_{tag}_{key[1]}", term, weight=w)

    def _add_diag_terms(self, out, batch, spatial_mask, lookback, t, w, loss, tag):
        for key, out_key in zip(self.diagnostic_keys, self.diagnostic_out_keys):
            target = batch[key][:, lookback + t, ...][spatial_mask]
            valid = torch.isfinite(target).all(dim=-1)
            if valid.sum() == 0:
                continue
            term = F.smooth_l1_loss(out[out_key][valid], target[valid], beta=1.0)
            loss.add(f"diag_{tag}_{key[1]}", term, weight=w)

    def _carry_and_update_history(self, out, batch, prog, history, spatial_mask, lookback, t, context_pairs):
        for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
            y_hat = prog[key] + out[out_key]
            prog[key] = y_hat.detach() if self.detach_rollout else y_hat
        self._update_history(prog, batch, history, spatial_mask, lookback, t, context_pairs)
        return prog

    # This loss implements its own __call__ and does not use the base per-step
    # hooks; these stubs exist only to satisfy the abstract base class.
    def _prognostic_loss(self, out, out_key, y_hat, target, valid, batch):
        raise NotImplementedError("RolloutIncrementLoss uses its own __call__, not the per-step hooks.")

    def _diagnostic_loss(self, out, out_key, target, valid, batch):
        raise NotImplementedError("RolloutIncrementLoss uses its own __call__, not the per-step hooks.")

    def _incremental_loss(self, out, out_key, pred_increment, target_increment, valid, batch):
        raise NotImplementedError("RolloutIncrementLoss uses its own __call__, not the per-step hooks.")
