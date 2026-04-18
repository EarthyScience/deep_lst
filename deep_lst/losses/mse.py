import torch
import torch.nn.functional as F
from tensordict import TensorDict


class DiagnosticMSE:
    def __init__(
        self,
        impute_inputs=True,  # single FVC values missing
        **kwargs,
    ):
        self.diagnostic_keys = kwargs["diagnostic_keys"]
        self.diagnostic_out_keys = kwargs["diagnostic_out_keys"]
        self.in_keys = kwargs.get("in_keys", [])
        self.impute_inputs = impute_inputs

    def __call__(self, lit_module, batch, stage: str):
        # Impute missing input values before forward pass
        if self.impute_inputs and self.in_keys:
            for key in self.in_keys:
                batch[key] = torch.nan_to_num(batch[key], nan=0.0)

        out = lit_module.model(batch)

        y_hat = out[self.diagnostic_out_keys[0]]
        y = batch[self.diagnostic_keys[0]]

        valid = torch.isfinite(y) & torch.isfinite(y_hat)
        count = valid.sum()
        lit_module.log(f"{stage}_count", count, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")

        if int(count.item()) == 0:
            print("no valid pixels found for loss computation!")
            loss = y_hat.sum() * 0.0
            lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            return loss

        loss = F.mse_loss(y_hat[valid], y[valid])
        lit_module.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss


class RolloutMSE:
    def __init__(
        self,
        detach_rollout=True,
        min_valid_fraction=0.05,
        incremental_loss=False,
        impute_inputs=True,  # single FVC values missing
        timestep_weights: list[float] | None = None,
        **kwargs,
    ):
        self.in_keys = kwargs["in_keys"]  # e.g. [("era5", "t2m"), ..., ("seviri", "DEM"), ...]
        self.prognostic_keys = kwargs["prognostic_keys"]  # e.g. [("seviri", "LST")]
        self.prognostic_out_keys = kwargs["prognostic_out_keys"]  # e.g. [("seviri", "LST")]
        self.diagnostic_keys = kwargs["diagnostic_keys"]  # e.g. [("seviri", "LST")]
        self.diagnostic_out_keys = kwargs["diagnostic_out_keys"]  # e.g. [("seviri", "LST")]
        self.detach_rollout = detach_rollout
        self.min_valid_fraction = min_valid_fraction
        self.incremental_loss = incremental_loss
        self.impute_inputs = impute_inputs
        self.timestep_weights = timestep_weights
        # Normalised increment variances: {var_name: (sigma_inc / sigma_var)^2}.
        # Divides each prognostic variable's loss by its increment variance so
        # gradient magnitude is comparable across variables regardless of how
        # slowly or quickly they change. Defaults to 1.0 (no scaling) when a
        # variable is absent — old statistics files fall through gracefully.
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
            state_0 = batch[key][:, lookback, ...]  # [B, H, W, C]
            mask = torch.isfinite(state_0).all(dim=-1)  # [B, H, W]
            spatial_mask = mask if spatial_mask is None else spatial_mask & mask

        if spatial_mask is None or spatial_mask.sum() == 0:
            print("  WARNING: No initially valid pixels!")
            return torch.tensor(0.0, device=batch.device, requires_grad=True)

        # Initialize prognostic state at t0 (batch index = lookback)
        prog = {}
        for key in self.prognostic_keys:
            prog[key] = batch[key][:, lookback, ...][spatial_mask]  # [N_valid, C]

        for t in range(1, horizon + 1):
            w = self.timestep_weights[t - 1] if self.timestep_weights is not None else 1.0

            # Build input TensorDict at valid pixels
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
                    valid_mask = torch.isfinite(ctx).all(dim=-1, keepdim=True)  # [N_valid, lookback, 1]
                    ctx_vals = torch.nan_to_num(ctx, nan=0.0)
                    # Flatten values and binary validity flag into one vector [N_valid, 2*lookback*C]
                    tstep_td[ctx_key] = torch.cat([ctx_vals.flatten(1), valid_mask.float().flatten(1)], dim=-1)

            out = lit_module.model(tstep_td)

            # Prognostic targets: model predicts increment
            for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
                pred_increment = out[out_key]
                y_hat = prog[key] + pred_increment
                target = batch[key][:, lookback + t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    sigma2_inc = self.inc_variances.get(key[1], 1.0)
                    var_mse = F.mse_loss(y_hat[target_valid], target[target_valid]) / sigma2_inc
                    loss += w * var_mse
                    var_losses[key[1]] += w * var_mse
                    var_weight_sums[key[1]] += w

                    if self.incremental_loss:
                        target_prev = batch[key][:, lookback + t - 1, ...][spatial_mask]
                        target_increment = target - target_prev
                        inc_valid = target_valid & torch.isfinite(target_increment).all(dim=-1)
                        if inc_valid.sum() > 0:
                            loss += w * F.mse_loss(pred_increment[inc_valid], target_increment[inc_valid]) / sigma2_inc

                    weight_sum += w

                # Update prognostic state (detach for standard rollout; keep graph for BPTT)
                new_prog = y_hat.detach() if self.detach_rollout else y_hat
                prog[key] = new_prog

            # Diagnostic targets: direct prediction at valid pixels
            for key, out_key in zip(self.diagnostic_keys, self.diagnostic_out_keys):
                y_hat = out[out_key]
                target = batch[key][:, lookback + t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    loss += w * F.mse_loss(y_hat[target_valid], target[target_valid])
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
