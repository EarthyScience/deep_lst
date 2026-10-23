import torch
import torch.nn.functional as F


def smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Huber loss with the knee at 1 (state is z-scored → 1 unit-std)."""
    return F.smooth_l1_loss(pred, target, beta=1.0)


LOSS_FNS = {"mse": F.mse_loss, "smooth_l1": smooth_l1, "mae": F.l1_loss}


def masked_loss(pred, target, loss_fn, min_valid_fraction):
    """Loss over entries where pred AND target are finite. Grad-safe zero if too few valid."""
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if mask.float().mean() < min_valid_fraction:
        return (torch.nan_to_num(pred) * 0.0).sum()
    return loss_fn(pred[mask], target[mask])


class RolloutLoss:
    def __init__(self, **kwargs):
        self.n_prognostic = len(kwargs["prognostic_keys"])
        self.detach_rollout = kwargs.get("detach_rollout", True)
        self.min_valid_fraction = kwargs.get("min_valid_fraction", 0.05)
        self.teacher_forcing_loss = kwargs.get("teacher_forcing_loss", True)
        self.incremental_loss = kwargs.get("incremental_loss", False)
        self.rollout_loss = kwargs.get("rollout_loss", True)
        # per-prognostic increment std σ_inc, used to standardise the increment term
        inc_variances = kwargs.get("inc_variances") or {}
        prognostic_vars = [key[1] for key in kwargs["prognostic_keys"]]
        self.inc_std = torch.tensor([inc_variances.get(var, 1.0) ** 0.5 for var in prognostic_vars])

        if not (self.teacher_forcing_loss or self.incremental_loss or self.rollout_loss):
            raise ValueError("RolloutLoss: all loss components disabled")

    def __call__(self, lit_module, batch, stage):
        model = lit_module.model
        cfg = lit_module.config["train_dataset"]["seviri"]
        lookback, horizon = cfg["lookback"], cfg["horizon"]
        loss_fn = LOSS_FNS[lit_module.config["model"].get("loss_fn", "mse")]

        y_prog = batch["y_prog"]  # (B, L+H+1, H, W, nS)  state, prog-first, NaNs kept
        X_dyn = batch["X_dyn"]  # (B, L+H+1, H, W, nF)  all inputs (step forcings)
        X_dyn_window = batch["X_dyn_ctx"]  # (B, L+H+1, H, W, nFc) forcing-context (window)

        n_states = y_prog.shape[-1]
        n_diagnostic = n_states - self.n_prognostic
        # Diagnostic-only run (no prognostic): the state and increment terms have nothing to
        # score, and the rollout carries nothing between steps, so it would only repeat the
        # teacher-forced diagnostic term.
        has_prognostic = self.n_prognostic > 0

        # increment term
        inc_std = self.inc_std.to(y_prog.device)
        target_inc = (
            y_prog[:, lookback + 1 : lookback + horizon + 1, ..., : self.n_prognostic]
            - y_prog[:, lookback : lookback + horizon, ..., : self.n_prognostic]
        )

        def state_loss(pred, target):
            return masked_loss(pred, target, loss_fn, self.min_valid_fraction)

        # ---------- teacher-forced branch ----------
        loss_logits = y_prog.new_zeros(())
        if self.teacher_forcing_loss or self.incremental_loss:
            y_prog_windows = []
            X_dyn_windows = []
            X_dyn_step = []
            for h in range(1, horizon + 1):
                y_prog_windows.append(
                    y_prog[:, h : lookback + h, ..., : self.n_prognostic]
                )  # (B, L, H, W, nP) true states
                X_dyn_windows.append(X_dyn_window[:, h : lookback + h])  # (B, L, H, W, nFc)
                X_dyn_step.append(X_dyn[:, lookback + h])  # (B, H, W, nF)
            y_prog_windows = torch.stack(y_prog_windows, dim=1)  # (B, H, L, H, W, nP)
            dyn_windows = torch.stack(X_dyn_windows, dim=1)
            step_dyn = torch.stack(X_dyn_step, dim=1)  # (B, H, H, W, nF)

            out = model(_fold(y_prog_windows), _fold(dyn_windows), _fold(step_dyn))  # (B*H, H, W, nS)
            out = out.reshape(*step_dyn.shape[:2], *out.shape[1:])  # (B, H, H, W, nS)

            inc_pred, diag_pred = out[..., : self.n_prognostic], out[..., self.n_prognostic :]

            prior_state = y_prog[
                :, lookback : lookback + horizon, ..., : self.n_prognostic
            ]  # (B, H, H, W, nP) true prior per lead
            target_state = y_prog[:, lookback + 1 : lookback + horizon + 1, ..., : self.n_prognostic]  # next step

            y_prog_pred = prior_state + inc_pred

            if self.teacher_forcing_loss:
                if has_prognostic:
                    loss_logits += state_loss(y_prog_pred, target_state)
                if n_diagnostic:
                    loss_logits += state_loss(
                        diag_pred, y_prog[:, lookback + 1 : lookback + horizon + 1, ..., self.n_prognostic :]
                    )

            if self.incremental_loss and has_prognostic:
                loss_logits += state_loss(inc_pred / inc_std, target_inc / inc_std)

        # ---------- rollout branch ----------
        if self.rollout_loss and has_prognostic:
            prog_seq = [y_prog[:, i, ..., : self.n_prognostic] for i in range(lookback + horizon + 1)]
            state_preds = []
            diag_preds = []
            inc_preds = []

            for h in range(1, horizon + 1):
                state_window = torch.stack(prog_seq[h : lookback + h], dim=1)  # (B, L, H, W, self.n_prognostic)
                out = model(state_window, X_dyn_window[:, h : lookback + h], X_dyn[:, lookback + h])  # (B, H, W, nS)

                inc = out[..., : self.n_prognostic]
                prior = prog_seq[lookback + h - 1]
                state_h = prior + inc

                state_preds.append(state_h)
                diag_preds.append(out[..., self.n_prognostic :])
                inc_preds.append(inc)

                if h < horizon:
                    prog_seq[lookback + h] = state_h.detach() if self.detach_rollout else state_h

            state_preds = torch.stack(state_preds, dim=1)  # (B, H, H, W, nP)
            loss_rollout = state_loss(
                state_preds, y_prog[:, lookback + 1 : lookback + horizon + 1, ..., : self.n_prognostic]
            )
            if n_diagnostic:
                diag_preds = torch.stack(diag_preds, dim=1)
                loss_rollout += state_loss(
                    diag_preds, y_prog[:, lookback + 1 : lookback + horizon + 1, ..., self.n_prognostic :]
                )
            if self.incremental_loss:
                inc_preds = torch.stack(inc_preds, dim=1)
                loss_rollout += state_loss(inc_preds / inc_std, target_inc / inc_std)
            loss_rollout = loss_rollout
        else:
            loss_rollout = y_prog.new_zeros(())

        total_loss = loss_logits + loss_rollout

        lit_module.log(f"{stage}_loss_step", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lit_module.log(f"{stage}_loss_logits", loss_logits, on_step=False, on_epoch=True, sync_dist=True)
        lit_module.log(f"{stage}_loss_rollout", loss_rollout, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss


def _fold(x):
    """(B, H, *rest) -> (B*H, *rest): fold the lead-time axis into the batch for one forward."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
