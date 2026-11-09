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
        # init_fill mirrors the inference key: where the training rollout takes its first prior.
        # none = the init observation (cloudy inits drop out of the rollout term); gated = the
        # observation where present, else the analysis; always = the analysis for every sample.
        # gated and always train the analysis: the diagnostics at t0 from the window ending at t0,
        # with the t0 slot hidden in init_hide_fraction of the samples so filling has a target.
        self.init_fill = kwargs.get("init_fill", "none")
        self.init_hide_fraction = kwargs.get("init_hide_fraction", 0.5)
        if self.init_fill not in ("none", "gated", "always"):
            raise ValueError(f"RolloutLoss: init_fill must be none, gated or always, got {self.init_fill!r}")
        prognostic_keys = kwargs["prognostic_keys"]
        diagnostic_keys = kwargs.get("diagnostic_keys", [])
        self.nowcast_channels = [
            (c, diagnostic_keys.index(key)) for c, key in enumerate(prognostic_keys) if key in diagnostic_keys
        ]  # (prognostic channel, diagnostic channel of its copy)
        if self.init_fill != "none" and len(self.nowcast_channels) < self.n_prognostic:
            raise ValueError("RolloutLoss: init_fill needs a diagnostic copy of every prognostic")
        # diagnostics whose observed past enters the window (diagnostic_history): those that are
        # not prognostics as well, whose history is in the window already
        self.history_channels = [d for d, key in enumerate(diagnostic_keys) if key not in prognostic_keys]
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

        # The window's state channels: the prognostics as they stand, and with diagnostic_history
        # the diagnostics' observed past, hidden from t0+1 on (they are predicted, never fed back).
        diagnostic_history = getattr(model, "diagnostic_history", True)
        history_prog = y_prog[..., : self.n_prognostic]
        history_diag = None
        if diagnostic_history and self.history_channels:
            history_diag = y_prog[..., self.n_prognostic :][..., self.history_channels].clone()
            history_diag[:, lookback + 1 :] = float("nan")

        # increment term
        inc_std = self.inc_std.to(y_prog.device)
        target_inc = (
            y_prog[:, lookback + 1 : lookback + horizon + 1, ..., : self.n_prognostic]
            - y_prog[:, lookback : lookback + horizon, ..., : self.n_prognostic]
        )

        def state_loss(pred, target):
            return masked_loss(pred, target, loss_fn, self.min_valid_fraction)

        # ---------- lead 0: the analysis and the start state ----------
        # gated / always: the diagnostics at t0 from the window ending at t0 (the init observation
        # inside it) and the forcings at t0, a geometry no lead h >= 1 covers, hence its own term;
        # the t0 slot is hidden in a random share of the samples so filling has a target. The start
        # state is the first prior of both branches: the observation where present (gated) or the
        # analysis for every sample (always).
        loss_analysis = y_prog.new_zeros(())
        start_state = None
        if self.init_fill != "none":
            window = history_prog[:, 1 : lookback + 1].clone()  # hours t0-L+1 .. t0
            if history_diag is not None:
                window = torch.cat([window, history_diag[:, 1 : lookback + 1]], dim=-1)
            if lookback > 0:
                hide = torch.rand(window.shape[0], device=window.device) < self.init_hide_fraction
                window[hide, -1] = float("nan")
            out0 = model(window, X_dyn_window[:, 1 : lookback + 1], X_dyn[:, lookback])
            analysis = out0[..., self.n_prognostic :]
            loss_analysis = state_loss(analysis, y_prog[:, lookback, ..., self.n_prognostic :])

            observed_init = y_prog[:, lookback, ..., : self.n_prognostic]
            start_state = observed_init.clone()
            for c, d in self.nowcast_channels:
                estimate = analysis[..., d]
                if self.init_fill == "gated":
                    estimate = torch.where(torch.isnan(observed_init[..., c]), estimate, observed_init[..., c])
                start_state[..., c] = estimate
            if self.detach_rollout:
                start_state = start_state.detach()

        # ---------- teacher-forced branch ----------
        loss_logits = y_prog.new_zeros(())
        if self.teacher_forcing_loss or self.incremental_loss:
            y_prog_windows = []
            X_dyn_windows = []
            X_dyn_step = []
            for h in range(1, horizon + 1):
                window = history_prog[:, h : lookback + h]  # (B, L, H, W, nP) true states
                if history_diag is not None:
                    window = torch.cat([window, history_diag[:, h : lookback + h]], dim=-1)
                y_prog_windows.append(window)
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
            if start_state is not None:  # lead 1 starts where the rollout starts
                prior_state = torch.cat([start_state.unsqueeze(1), prior_state[:, 1:]], dim=1)
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
            if start_state is not None:
                prog_seq[lookback] = start_state
            state_preds = []
            diag_preds = []
            inc_preds = []

            for h in range(1, horizon + 1):
                state_window = torch.stack(prog_seq[h : lookback + h], dim=1)  # (B, L, H, W, nP)
                if history_diag is not None:
                    state_window = torch.cat([state_window, history_diag[:, h : lookback + h]], dim=-1)
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

        total_loss = loss_logits + loss_rollout + loss_analysis

        lit_module.log(f"{stage}_loss_step", total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        lit_module.log(f"{stage}_loss_logits", loss_logits, on_step=False, on_epoch=True, sync_dist=True)
        lit_module.log(f"{stage}_loss_rollout", loss_rollout, on_step=False, on_epoch=True, sync_dist=True)
        lit_module.log(f"{stage}_loss_analysis", loss_analysis, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss


def _fold(x):
    """(B, H, *rest) -> (B*H, *rest): fold the lead-time axis into the batch for one forward."""
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
