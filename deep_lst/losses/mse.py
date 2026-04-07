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

    def __call__(self, lit_module, batch, stage: str):
        horizon = lit_module.config["train_dataset"]["seviri"]["horizon"]  # suboptimal
        #lookback = lit_module.config["train_dataset"]["seviri"]["lookback"]

        loss = torch.tensor(0.0, device=batch.device)
        valid_timesteps = 0

        # Compute combined spatial validity mask from initial prognostic state (t=0)
        # Reduce across channels (all channels must be valid per pixel)
        spatial_mask = None
        for key in self.prognostic_keys:
            state_0 = batch[key][:, 0, ...]  # [B, H, W, C]
            mask = torch.isfinite(state_0).all(dim=-1)  # [B, H, W]
            spatial_mask = mask if spatial_mask is None else spatial_mask & mask

        if spatial_mask is None or spatial_mask.sum() == 0:
            print("  WARNING: No initially valid pixels!")
            return torch.tensor(0.0, device=batch.device, requires_grad=True)

        # Subset to valid pixels: [B, H, W, ...] -> [N_valid, ...]
        # Initialize prognostic state at valid pixels only
        prog = {}
        for key in self.prognostic_keys:
            prog[key] = batch[key][:, 0, ...][spatial_mask]  # [N_valid, C]

        for t in range(1, horizon + 1):
            # Build input at valid pixels only
            tstep_td = TensorDict({}, batch_size=[spatial_mask.sum().item()])
            for key in self.in_keys:
                inp = batch[key][:, t, ...][spatial_mask]
                if self.impute_inputs:
                    inp = torch.nan_to_num(inp, nan=0.0)
                tstep_td[key] = inp
            for key in self.prognostic_keys:
                tstep_td[key] = prog[key]

            out = lit_module.model(tstep_td)

            # Prognostic targets: model predicts increment
            for key, out_key in zip(self.prognostic_keys, self.prognostic_out_keys):
                pred_increment = out[out_key]
                y_hat = prog[key] + pred_increment
                target = batch[key][:, t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    # State loss
                    loss += F.mse_loss(y_hat[target_valid], target[target_valid])

                    # Increment loss: compare predicted increment to observed increment
                    if self.incremental_loss:
                        target_prev = batch[key][:, t - 1, ...][spatial_mask]
                        target_increment = target - target_prev
                        inc_valid = target_valid & torch.isfinite(target_increment).all(dim=-1)
                        if inc_valid.sum() > 0:
                            loss += F.mse_loss(pred_increment[inc_valid], target_increment[inc_valid])

                    valid_timesteps += 1

                # Update prognostic state
                new_prog = y_hat.detach() if self.detach_rollout else y_hat
                prog[key] = new_prog

            # Diagnostic targets: direct prediction at valid pixels
            for key, out_key in zip(self.diagnostic_keys, self.diagnostic_out_keys):
                y_hat = out[out_key]
                target = batch[key][:, t, ...][spatial_mask]
                target_valid = torch.isfinite(target).all(dim=-1)

                frac = target_valid.sum().float() / max(target_valid.numel(), 1)
                if frac >= self.min_valid_fraction:
                    loss += F.mse_loss(y_hat[target_valid], target[target_valid])
                    valid_timesteps += 1

        if valid_timesteps > 0:
            loss = loss / valid_timesteps

        lit_module.log(f"{stage}_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lit_module.log(f"{stage}_valid_timesteps", float(valid_timesteps), sync_dist=True)

        return loss
