"""Tests for deep_lst/losses/rollout.py

Hermetic: a stub model that returns one learnable value per output channel.

Covered:
  RolloutLoss with zero prognostics (diagnostic-only run, v18 no. 4)
    - the loss is finite and equals the teacher-forced diagnostic term
    - the rollout term is zero (nothing is carried between steps)
    - gradients reach the model
  RolloutLoss with one prognostic
    - the state and rollout terms stay active (guard for the nP > 0 path)
    - the total is pinned to its value before the diagnostic-history change
  diagnostic_history
    - the diagnostics' window slots after t0 are hidden at every lead
  init_fill gated / always (joint head: LST prognostic and diagnostic)
    - the lead-0 term scores the window ending at t0 with its t0 slot hidden at the given share
    - gated starts the rollout from the observation where present, else from the analysis
    - always starts every sample from the analysis
    - init_fill is validated at construction
    - a diagnostic that is also a prognostic adds no window channel
"""

import pytest
import torch
import torch.nn.functional as F

from deep_lst.losses import RolloutLoss

B = 3
LOOKBACK = 4
HORIZON = 2
T = LOOKBACK + HORIZON + 1
N_FORCING = 2
N_CONTEXT = 1


class _ConstantModel(torch.nn.Module):
    """Returns one learnable value per output channel, broadcast over the batch."""

    def __init__(self, n_out):
        super().__init__()
        self.values = torch.nn.Parameter(torch.linspace(0.1, 0.5, n_out))

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        N, H, W, _ = step_X_dyn.shape
        return self.values.expand(N, H, W, -1)


class _StubModule:
    """The two LitModelModule attributes RolloutLoss reads, plus a log() that records values."""

    def __init__(self, model):
        self.model = model
        self.config = {
            "train_dataset": {"seviri": {"lookback": LOOKBACK, "horizon": HORIZON}},
            "model": {"loss_fn": "smooth_l1"},
        }
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value.detach())


def _batch(n_states, seed=0):
    generator = torch.Generator().manual_seed(seed)
    y_prog = torch.randn(B, T, 1, 1, n_states, generator=generator)
    y_prog[0, LOOKBACK + 1, 0, 0, :] = float("nan")  # one masked target
    X_dyn = torch.randn(B, T, 1, 1, N_FORCING, generator=generator)
    X_dyn_ctx = torch.randn(B, T, 1, 1, N_CONTEXT, generator=generator)
    return {"y_prog": y_prog, "X_dyn": X_dyn, "X_dyn_ctx": X_dyn_ctx}


def _loss(prognostic_keys, **extra):
    return RolloutLoss(
        prognostic_keys=prognostic_keys,
        detach_rollout=False,
        teacher_forcing_loss=True,
        incremental_loss=False,
        rollout_loss=True,
        **extra,
    )


def test_zero_prognostics_gives_the_diagnostic_term_only():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)
    batch = _batch(n_states=1)

    total = _loss(prognostic_keys=[])(module, batch, stage="train")

    targets = batch["y_prog"][:, LOOKBACK + 1 :, ..., 0]
    valid = torch.isfinite(targets)
    expected = F.smooth_l1_loss(model.values[0].expand_as(targets)[valid], targets[valid])
    assert torch.isfinite(total)
    assert torch.isclose(total, expected)
    assert module.logged["train_loss_rollout"] == 0.0


def test_zero_prognostics_backpropagates():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)

    total = _loss(prognostic_keys=[])(module, _batch(n_states=1), stage="train")
    total.backward()

    assert torch.isfinite(model.values.grad).all()


def test_one_prognostic_keeps_state_and_rollout_terms():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)

    total = _loss(prognostic_keys=[("seviri", "LST")])(module, _batch(n_states=1), stage="train")

    assert torch.isfinite(total)
    assert module.logged["train_loss_logits"] > 0.0
    assert module.logged["train_loss_rollout"] > 0.0
    assert torch.isclose(total, torch.tensor(1.45839787), atol=1e-6)  # value before diagnostic_history


class _RecordingModel(_ConstantModel):
    """Keeps every state window it was called with."""

    def __init__(self, n_out, diagnostic_history):
        super().__init__(n_out)
        self.diagnostic_history = diagnostic_history
        self.windows = []

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        self.windows.append(state_window)
        return super().forward(state_window, X_dyn_window, step_X_dyn)


def test_diagnostic_history_hides_the_future_in_the_window():
    model = _RecordingModel(n_out=1, diagnostic_history=True)
    module = _StubModule(model)

    _loss(prognostic_keys=[], diagnostic_keys=[("seviri", "LST")])(module, _batch(n_states=1), stage="train")

    window = model.windows[0].reshape(B, HORIZON, LOOKBACK, 1, 1, 1)  # teacher-forced fold undone
    for h in range(1, HORIZON + 1):
        visible = window[:, h - 1, : LOOKBACK - h + 1]  # hours up to t0
        hidden = window[:, h - 1, LOOKBACK - h + 1 :]  # hours after t0
        assert torch.isfinite(visible).all()
        assert torch.isnan(hidden).all()


def test_without_diagnostic_history_the_window_has_prognostics_only():
    model = _RecordingModel(n_out=2, diagnostic_history=False)
    module = _StubModule(model)

    _loss(prognostic_keys=[("seviri", "LST")])(module, _batch(n_states=2), stage="train")

    assert all(window.shape[-1] == 1 for window in model.windows)


JOINT_KEYS = {"prognostic_keys": [("seviri", "LST")], "diagnostic_keys": [("seviri", "LST")]}


def test_nowcast_loss_scores_the_window_ending_at_t0_with_its_t0_slot_hidden():
    model = _RecordingModel(n_out=2, diagnostic_history=False)
    module = _StubModule(model)
    batch = _batch(n_states=2)

    _loss(init_fill="gated", init_hide_fraction=1.0, **JOINT_KEYS)(module, batch, stage="train")

    assert module.logged["train_loss_analysis"] > 0.0
    nowcast_window = model.windows[0]  # call 0 = the analysis, before the teacher-forced fold
    assert nowcast_window.shape == (B, LOOKBACK, 1, 1, 1)
    assert torch.isnan(nowcast_window[:, -1]).all()  # every t0 slot hidden at fraction 1.0
    assert torch.equal(nowcast_window[:, :-1], batch["y_prog"][:, 1:LOOKBACK, ..., :1])  # hours t0-L+1 .. t0-1


def test_nowcast_hide_fraction_zero_keeps_the_init_observation():
    model = _RecordingModel(n_out=2, diagnostic_history=False)
    module = _StubModule(model)
    batch = _batch(n_states=2)

    _loss(init_fill="gated", init_hide_fraction=0.0, **JOINT_KEYS)(module, batch, stage="train")

    assert torch.equal(model.windows[0], batch["y_prog"][:, 1 : LOOKBACK + 1, ..., :1])


def test_init_fill_always_starts_every_sample_from_the_analysis():
    model = _RecordingModel(n_out=2, diagnostic_history=False)  # values [0.1, 0.5]: analysis = 0.5
    module = _StubModule(model)

    _loss(init_fill="always", init_hide_fraction=0.0, **JOINT_KEYS)(module, _batch(n_states=2), stage="train")

    first_rollout_window = model.windows[2]  # call 2 = rollout lead 1, window hours t0-L+1 .. t0
    assert torch.all(first_rollout_window[:, -1, ..., 0] == 0.5)  # the t0 slot holds the analysis


def test_init_fill_gated_starts_from_the_observation_where_present():
    model = _RecordingModel(n_out=2, diagnostic_history=False)  # analysis = 0.5
    module = _StubModule(model)
    batch = _batch(n_states=2)
    batch["y_prog"][0, LOOKBACK] = float("nan")  # sample 0 has no init observation

    _loss(init_fill="gated", init_hide_fraction=0.0, **JOINT_KEYS)(module, batch, stage="train")

    start = model.windows[2][:, -1, ..., 0]  # rollout lead 1, the t0 slot
    assert torch.all(start[0] == 0.5)
    assert torch.equal(start[1:], batch["y_prog"][1:, LOOKBACK, ..., 0])
    assert module.logged["train_loss_rollout"] > 0.0


def test_a_diagnostic_that_is_also_a_prognostic_adds_no_window_channel():
    model = _RecordingModel(n_out=2, diagnostic_history=True)
    module = _StubModule(model)

    _loss(**JOINT_KEYS)(module, _batch(n_states=2), stage="train")

    assert all(window.shape[-1] == 1 for window in model.windows)


def test_init_fill_is_validated():
    with pytest.raises(ValueError, match="none, gated or always"):
        _loss(init_fill="sometimes", **JOINT_KEYS)
    with pytest.raises(ValueError, match="diagnostic copy"):
        _loss(prognostic_keys=[("seviri", "LST")], init_fill="gated")
