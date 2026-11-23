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
  terms
    - a weight multiplies its term; rollout 0 skips the rollout branch
    - all weights 0 is refused
  rollout (the free-running forecast, called on its own)
    - the model is called once per lead with lookback-long windows of the prognostics
    - a gap in the observations stays a gap in the window
    - one prediction per lead comes back: states, diagnostics, increments
    - a prediction is valid where the init observation is, whatever the targets
  masked_loss
    - scores the entries that are valid in prediction and target
"""

import pytest
import torch
import torch.nn.functional as F

from deep_lst.losses import RolloutLoss
from deep_lst.losses.rollout import masked_loss

B = 3
LOOKBACK = 4
HORIZON = 2
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
        self.config = {"window": {"lookback": LOOKBACK, "horizon": HORIZON}}
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value.detach())


def _batch(n_states, seed=0, horizon=HORIZON, height=1, width=1):
    generator = torch.Generator().manual_seed(seed)
    n_steps = LOOKBACK + horizon + 1
    y_prog = torch.randn(B, n_steps, height, width, n_states, generator=generator)
    y_prog[0, LOOKBACK + 1, 0, 0, :] = float("nan")  # one masked target
    X_dyn = torch.randn(B, n_steps, height, width, N_FORCING, generator=generator)
    X_dyn_ctx = torch.randn(B, n_steps, height, width, N_CONTEXT, generator=generator)
    return {"y_prog": y_prog, "X_dyn": X_dyn, "X_dyn_ctx": X_dyn_ctx}


def _loss(prognostic_keys, **extra):
    """RolloutLoss with the values of conf/objective/rollout.yaml and loss=smoothl1."""
    settings = {
        "pointwise": F.smooth_l1_loss,
        "terms": {"teacher_forcing": 1.0, "rollout": 1.0, "increment": 0.0},
        "detach_rollout": False,
        "min_valid_fraction": 0.05,
        "init_fill": "none",
        "init_hide_fraction": 0.5,
        "kind": "rollout",
        "checkpoint_monitor": "val_loss_step_epoch_log",
    }
    settings.update(extra)
    return RolloutLoss(prognostic_keys=prognostic_keys, **settings)


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
    """Keeps the inputs of every call."""

    def __init__(self, n_out, diagnostic_history):
        super().__init__(n_out)
        self.diagnostic_history = diagnostic_history
        self.windows = []
        self.forcing_windows = []
        self.step_forcings = []

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        self.windows.append(state_window)
        self.forcing_windows.append(X_dyn_window)
        self.step_forcings.append(step_X_dyn)
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


def test_a_weight_multiplies_its_term():
    keys = [("seviri", "LST")]
    reference = _StubModule(_ConstantModel(n_out=1))
    _loss(prognostic_keys=keys)(reference, _batch(n_states=1), stage="train")
    weighted = _StubModule(_ConstantModel(n_out=1))
    terms = {"teacher_forcing": 2.0, "rollout": 0.5, "increment": 0.0}

    total = _loss(prognostic_keys=keys, terms=terms)(weighted, _batch(n_states=1), stage="train")

    assert weighted.logged["train_loss_logits"] == pytest.approx(2.0 * reference.logged["train_loss_logits"])
    assert weighted.logged["train_loss_rollout"] == pytest.approx(0.5 * reference.logged["train_loss_rollout"])
    expected = weighted.logged["train_loss_logits"] + weighted.logged["train_loss_rollout"]
    assert float(total.detach()) == pytest.approx(expected)


def test_rollout_weight_zero_skips_the_rollout_branch():
    model = _RecordingModel(n_out=1, diagnostic_history=False)
    module = _StubModule(model)
    terms = {"teacher_forcing": 1.0, "rollout": 0.0, "increment": 0.0}

    _loss(prognostic_keys=[("seviri", "LST")], terms=terms)(module, _batch(n_states=1), stage="train")

    assert len(model.windows) == 1  # the folded teacher-forced forward only
    assert module.logged["train_loss_rollout"] == 0.0


def test_all_weights_zero_is_refused():
    terms = {"teacher_forcing": 0.0, "rollout": 0.0, "increment": 0.0}
    with pytest.raises(ValueError, match="weight 0"):
        _loss(prognostic_keys=[("seviri", "LST")], terms=terms)


# ---------- rollout, called on its own ----------
# Sizes differ per axis (B 3, LOOKBACK 4, HEIGHT 5, WIDTH 6), so a swapped axis cannot pass.
HEIGHT = 5
WIDTH = 6
ROLLOUT_KEYS = [("seviri", "LST"), ("era5_land", "skt")]
N_PROGNOSTIC = len(ROLLOUT_KEYS)
N_DIAGNOSTIC = 1
N_STATES = N_PROGNOSTIC + N_DIAGNOSTIC
HORIZONS = [1, 2, 7]


def _run_rollout(model, batch, horizon):
    loss = _loss(prognostic_keys=ROLLOUT_KEYS)
    return loss.rollout(model, batch["y_prog"], batch["X_dyn"], batch["X_dyn_ctx"], LOOKBACK, horizon)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_rollout_calls_the_model_once_per_lead_with_a_lookback_window(horizon):
    model = _RecordingModel(n_out=N_STATES, diagnostic_history=False)
    batch = _batch(n_states=N_STATES, horizon=horizon, height=HEIGHT, width=WIDTH)

    _run_rollout(model, batch, horizon)

    assert len(model.windows) == horizon
    for lead in range(horizon):
        assert model.windows[lead].shape == (B, LOOKBACK, HEIGHT, WIDTH, N_PROGNOSTIC)
        assert model.forcing_windows[lead].shape == (B, LOOKBACK, HEIGHT, WIDTH, N_CONTEXT)
        assert model.step_forcings[lead].shape == (B, HEIGHT, WIDTH, N_FORCING)


def test_rollout_window_keeps_the_gaps_of_the_observations():
    model = _RecordingModel(n_out=N_STATES, diagnostic_history=False)
    batch = _batch(n_states=N_STATES, height=HEIGHT, width=WIDTH)
    batch["y_prog"][1, 2, 3, 4, 0] = float("nan")  # one gap inside the lookback

    _run_rollout(model, batch, HORIZON)

    observed_window = batch["y_prog"][:, 1 : LOOKBACK + 1, ..., :N_PROGNOSTIC]  # hours t0-L+1 .. t0
    assert torch.isnan(model.windows[0][1, 1, 3, 4, 0])  # time step 2 is window slot 1
    assert torch.equal(torch.isnan(model.windows[0]), torch.isnan(observed_window))


@pytest.mark.parametrize("horizon", HORIZONS)
def test_rollout_returns_one_prediction_per_lead(horizon):
    model = _RecordingModel(n_out=N_STATES, diagnostic_history=False)
    batch = _batch(n_states=N_STATES, horizon=horizon, height=HEIGHT, width=WIDTH)

    state_preds, diag_preds, inc_preds = _run_rollout(model, batch, horizon)

    assert state_preds.shape == (B, horizon, HEIGHT, WIDTH, N_PROGNOSTIC)
    assert diag_preds.shape == (B, horizon, HEIGHT, WIDTH, N_DIAGNOSTIC)
    assert inc_preds.shape == (B, horizon, HEIGHT, WIDTH, N_PROGNOSTIC)


def test_rollout_prediction_is_valid_where_the_init_observation_is():
    horizon = 7
    model = _RecordingModel(n_out=N_STATES, diagnostic_history=False)
    batch = _batch(n_states=N_STATES, horizon=horizon, height=HEIGHT, width=WIDTH)
    batch["y_prog"][0, LOOKBACK, 1, 1, 0] = float("nan")  # no init observation: every lead is lost
    batch["y_prog"][2, LOOKBACK + 3, 0, 0, 1] = float("nan")  # a missing target: the prediction stays

    state_preds, _, _ = _run_rollout(model, batch, horizon)

    init_valid = torch.isfinite(batch["y_prog"][:, LOOKBACK, ..., :N_PROGNOSTIC])  # (B, HEIGHT, WIDTH, nP)
    expected_valid = init_valid.unsqueeze(1).expand_as(state_preds)
    assert torch.isnan(state_preds[0, :, 1, 1, 0]).all()  # the pixel without init observation, at every lead
    assert torch.equal(torch.isfinite(state_preds), expected_valid)


def test_masked_loss_scores_the_entries_valid_in_prediction_and_target():
    pred = torch.zeros(B, HORIZON, HEIGHT, WIDTH, N_PROGNOSTIC)
    target = torch.zeros(B, HORIZON, HEIGHT, WIDTH, N_PROGNOSTIC)
    pred[0, 0, 1, 1, 0] = float("nan")
    target[2, 1, 0, 0, 1] = float("nan")
    scored_sizes = []

    def recording_pointwise(valid_pred, valid_target):
        scored_sizes.append((valid_pred.numel(), valid_target.numel()))
        return valid_pred.sum()

    masked_loss(pred, target, recording_pointwise, min_valid_fraction=0.05)

    n_valid = pred.numel() - 2
    assert scored_sizes == [(n_valid, n_valid)]
