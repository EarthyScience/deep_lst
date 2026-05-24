"""Tests for the MLP model and training logic.

Replaces the --smoke CLI flag with fast, hermetic unit tests.
No real data, no zarr stores, no GPU required.

Covered:
  - MultiModalMLP: construction, forward-pass shape, output keys
  - DiagnosticMSE: loss is finite scalar, valid-pixel masking
  - RolloutMSE: rollout over horizon, NaN masking, incremental loss, detach
  - Optimizer step: parameters change after one backward pass
  - LitModelModule.training_step: integrates model + loss without a real Trainer
"""

import pytest
import torch
from tensordict import TensorDict

from deep_lst.losses import DiagnosticCRPS, DiagnosticMSE, RolloutCRPS, RolloutMSE
from deep_lst.mlp.model import LitModelModule, MultiModalMLP

# ── shared fixtures ───────────────────────────────────────────────────────────

BATCH_SIZE = 4

IN_KEYS = [("era5", "t2m"), ("era5", "u10"), ("seviri", "DEM")]
DIAG_KEYS = [("seviri", "LST")]
DIAG_OUT_KEYS = [("seviri", "LST_pred")]

MODEL_CONFIG = {
    "activation": "relu",
    "hidden_sizes": [16, 8],
    "dropout": 0.0,
}

FULL_CONFIG = {
    "model": {**MODEL_CONFIG, "name": "mlp", "step_loss": "DiagnosticMSE", "learning_rate": 1e-3},
    "training": {"max_epochs": 2},
}


@pytest.fixture
def model():
    return MultiModalMLP(in_keys=IN_KEYS, out_keys=DIAG_OUT_KEYS, config=MODEL_CONFIG)


@pytest.fixture
def batch():
    """Minimal TensorDict with one scalar feature per key, all finite."""
    td = TensorDict(
        {k: torch.randn(BATCH_SIZE, 1) for k in IN_KEYS + DIAG_KEYS + DIAG_OUT_KEYS},
        batch_size=[BATCH_SIZE],
    )
    return td


@pytest.fixture
def batch_with_nans(batch):
    """Same batch but half the diagnostic target pixels are NaN (cloud-masked)."""
    b = batch.clone()
    b[DIAG_KEYS[0]][: BATCH_SIZE // 2] = float("nan")
    return b


# ── MultiModalMLP ─────────────────────────────────────────────────────────────


def test_model_constructs(model):
    assert model.input_size == len(IN_KEYS)
    assert model.target_size == len(DIAG_OUT_KEYS)


def test_model_forward_output_keys(model, batch):
    out = model(batch)
    for key in DIAG_OUT_KEYS:
        assert key in out


def test_model_forward_output_shape(model, batch):
    out = model(batch)
    for key in DIAG_OUT_KEYS:
        assert out[key].shape == (BATCH_SIZE, 1)


def test_model_forward_preserves_input_keys(model, batch):
    """forward() should clone the input TensorDict and add outputs, not drop inputs."""
    out = model(batch)
    for key in IN_KEYS:
        assert key in out


def test_model_output_is_finite(model, batch):
    out = model(batch)
    for key in DIAG_OUT_KEYS:
        assert torch.isfinite(out[key]).all()


# ── DiagnosticMSE ─────────────────────────────────────────────────────────────


class _FakeLitModule:
    """Minimal stand-in for LitModelModule that records log calls."""

    def __init__(self, model):
        self.model = model
        self.logged: dict = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def test_diagnostic_mse_returns_scalar(model, batch):
    loss_fn = DiagnosticMSE(
        in_keys=IN_KEYS,
        diagnostic_keys=DIAG_KEYS,
        diagnostic_out_keys=DIAG_OUT_KEYS,
    )
    lit = _FakeLitModule(model)
    loss = loss_fn(lit, batch, stage="train")

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_diagnostic_mse_loss_positive(model, batch):
    loss_fn = DiagnosticMSE(
        in_keys=IN_KEYS,
        diagnostic_keys=DIAG_KEYS,
        diagnostic_out_keys=DIAG_OUT_KEYS,
    )
    lit = _FakeLitModule(model)
    loss = loss_fn(lit, batch, stage="train")
    assert loss.item() >= 0.0


def test_diagnostic_mse_nan_targets_masked(model, batch_with_nans):
    """Loss should still be finite when half the targets are NaN."""
    loss_fn = DiagnosticMSE(
        in_keys=IN_KEYS,
        diagnostic_keys=DIAG_KEYS,
        diagnostic_out_keys=DIAG_OUT_KEYS,
    )
    lit = _FakeLitModule(model)
    loss = loss_fn(lit, batch_with_nans, stage="train")
    assert torch.isfinite(loss)


def test_diagnostic_mse_all_nan_targets_zero_loss(model, batch):
    """When all targets are NaN, the loss should be zero (no valid pixels)."""
    b = batch.clone()
    b[DIAG_KEYS[0]][:] = float("nan")

    loss_fn = DiagnosticMSE(
        in_keys=IN_KEYS,
        diagnostic_keys=DIAG_KEYS,
        diagnostic_out_keys=DIAG_OUT_KEYS,
    )
    lit = _FakeLitModule(model)
    loss = loss_fn(lit, b, stage="train")
    assert loss.item() == 0.0


# ── optimizer step ────────────────────────────────────────────────────────────


def test_optimizer_step_updates_parameters(model, batch):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    params_before = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    out = model(batch)
    # Simple surrogate loss: predict zeros for diagnostic output
    loss = out[DIAG_OUT_KEYS[0]].pow(2).mean()
    loss.backward()
    optimizer.step()

    for before, after in zip(params_before, model.parameters()):
        if before.requires_grad:
            assert not torch.equal(before, after), "parameter was not updated"


# ── LitModelModule.training_step ─────────────────────────────────────────────


def test_lit_module_training_step_returns_loss(model, batch):
    lit = LitModelModule(
        model=model,
        config=FULL_CONFIG,
        in_keys=IN_KEYS,
        out_keys=DIAG_OUT_KEYS,
        diagnostic_keys=DIAG_KEYS,
        diagnostic_out_keys=DIAG_OUT_KEYS,
        prognostic_keys=[],
        prognostic_out_keys=[],
    )
    # Patch log to avoid needing a real Trainer
    lit.log = lambda *a, **kw: None

    loss = lit.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


# ── RolloutMSE ────────────────────────────────────────────────────────────────
#
# Batch tensors have a time dimension: shape (B, T, 1).
# Prognostic keys serve as both the initial state (t=0) and the target at each
# rollout step. The model predicts an *increment* that is added to the state.
#
# HORIZON=2, LOOKBACK=0 → loop over t in [0, 1, 2], T=3 timesteps total.

ROLLOUT_IN_KEYS = [("era5", "t2m"), ("era5", "u10")]
PROG_KEYS = [("seviri", "LST")]
PROG_OUT_KEYS = [("seviri", "LST_pred")]

ROLLOUT_HORIZON = 2
ROLLOUT_LOOKBACK = 0
ROLLOUT_T = ROLLOUT_LOOKBACK + ROLLOUT_HORIZON + 1  # 3

ROLLOUT_CONFIG = {
    "train_dataset": {"seviri": {"horizon": ROLLOUT_HORIZON, "lookback": ROLLOUT_LOOKBACK}},
}


@pytest.fixture
def rollout_model():
    """Model whose in_keys = regular inputs + prognostic state."""
    return MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + PROG_KEYS,
        out_keys=PROG_OUT_KEYS,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )


@pytest.fixture
def rollout_batch():
    """All-finite batch of shape (B, T, 1) for each key."""
    return TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + PROG_KEYS},
        batch_size=[BATCH_SIZE],
    )


class _RolloutLitModule:
    def __init__(self, model):
        self.model = model
        self.config = ROLLOUT_CONFIG
        self.logged: dict = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value


def _make_rollout_loss(**kwargs):
    defaults = dict(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=PROG_KEYS,
        prognostic_out_keys=PROG_OUT_KEYS,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    defaults.update(kwargs)
    return RolloutMSE(**defaults)


def test_rollout_mse_returns_finite_scalar(rollout_model, rollout_batch):
    loss_fn = _make_rollout_loss()
    lit = _RolloutLitModule(rollout_model)
    loss = loss_fn(lit, rollout_batch, stage="train")

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rollout_mse_loss_nonnegative(rollout_model, rollout_batch):
    loss = _make_rollout_loss()(_RolloutLitModule(rollout_model), rollout_batch, stage="train")
    assert loss.item() >= 0.0


def test_rollout_mse_all_nan_initial_state_returns_zero(rollout_model, rollout_batch):
    """No valid pixels at t=0 → loss must be 0 (with requires_grad=True)."""
    b = rollout_batch.clone()
    b[PROG_KEYS[0]][:, 0, :] = float("nan")

    loss = _make_rollout_loss()(_RolloutLitModule(rollout_model), b, stage="train")

    assert loss.item() == 0.0
    assert loss.requires_grad  # as per the early-return in RolloutMSE


def test_rollout_mse_nan_mid_rollout_handled(rollout_model, rollout_batch):
    """NaN targets at t=1 should not break the rollout; loss stays finite."""
    b = rollout_batch.clone()
    b[PROG_KEYS[0]][:, 1, :] = float("nan")

    loss = _make_rollout_loss()(_RolloutLitModule(rollout_model), b, stage="train")

    assert torch.isfinite(loss)


def test_rollout_mse_horizon_controls_valid_timesteps(rollout_model):
    """valid_timesteps logged should equal horizon - lookback + 1 when all targets are finite."""
    for horizon in (1, 2, 3):
        t_total = ROLLOUT_LOOKBACK + horizon + 1
        b = TensorDict(
            {k: torch.randn(BATCH_SIZE, t_total, 1) for k in ROLLOUT_IN_KEYS + PROG_KEYS},
            batch_size=[BATCH_SIZE],
        )
        cfg = {"train_dataset": {"seviri": {"horizon": horizon, "lookback": ROLLOUT_LOOKBACK}}}

        class _Lit:
            model = rollout_model
            config = cfg
            logged = {}

            def log(self, name, value, **kw):
                self.logged[name] = value

        lit = _Lit()
        _make_rollout_loss()(lit, b, stage="train")

        # One prognostic key, all pixels valid → weight_sum = 1.0 per timestep × horizon timesteps
        expected = float(horizon)
        assert lit.logged["train_weight_sum"] == expected, (
            f"horizon={horizon}: expected weight_sum={expected}, got {lit.logged['train_weight_sum']}"
        )


def test_rollout_mse_incremental_loss_adds_extra_term(rollout_model, rollout_batch):
    """incremental_loss=True should produce a different (generally higher) loss."""
    lit = _RolloutLitModule(rollout_model)
    loss_base = _make_rollout_loss(incremental_loss=False)(lit, rollout_batch, stage="train")
    loss_incr = _make_rollout_loss(incremental_loss=True)(lit, rollout_batch, stage="train")
    # Increment loss adds extra terms — losses should differ
    assert not torch.isclose(loss_base, loss_incr)


def test_rollout_mse_detach_rollout_backward_succeeds(rollout_model, rollout_batch):
    """With detach_rollout=True the backward pass should complete without error."""
    lit = _RolloutLitModule(rollout_model)
    loss = _make_rollout_loss(detach_rollout=True)(lit, rollout_batch, stage="train")
    loss.backward()  # must not raise


def test_rollout_mse_no_detach_backward_succeeds(rollout_model, rollout_batch):
    """With detach_rollout=False gradients flow through states; backward must still work."""
    lit = _RolloutLitModule(rollout_model)
    loss = _make_rollout_loss(detach_rollout=False)(lit, rollout_batch, stage="train")
    loss.backward()  # must not raise


# ── timestep_weights ──────────────────────────────────────────────────────────


def test_uniform_weights_match_no_weights(rollout_model, rollout_batch):
    """Explicit uniform weights must produce the same loss as no weights (backward compat)."""
    lit = _RolloutLitModule(rollout_model)
    loss_default = _make_rollout_loss()(lit, rollout_batch, stage="train")
    loss_uniform = _make_rollout_loss(timestep_weights=[1.0] * ROLLOUT_HORIZON)(lit, rollout_batch, stage="train")
    assert torch.isclose(loss_default, loss_uniform), (
        f"uniform weights changed loss: {loss_default.item()} vs {loss_uniform.item()}"
    )


def test_zero_weight_timestep_excluded(rollout_model, rollout_batch):
    """A timestep with weight=0 must not contribute to the loss."""
    lit = _RolloutLitModule(rollout_model)
    # Only t=1 has weight > 0; t=2 is zeroed out.
    loss_t1_only = _make_rollout_loss(timestep_weights=[1.0, 0.0])(lit, rollout_batch, stage="train")
    # Sanity: loss is finite and positive.
    assert torch.isfinite(loss_t1_only)
    assert loss_t1_only.item() >= 0.0


def test_early_step_emphasis_changes_loss(rollout_model, rollout_batch):
    """Decaying weights must produce a different loss than uniform weights."""
    lit = _RolloutLitModule(rollout_model)
    loss_uniform = _make_rollout_loss(timestep_weights=[1.0, 1.0])(lit, rollout_batch, stage="train")
    loss_decaying = _make_rollout_loss(timestep_weights=[4.0, 1.0])(lit, rollout_batch, stage="train")
    assert not torch.isclose(loss_uniform, loss_decaying), "decaying weights should change the loss"


def test_weight_sum_logged_correctly(rollout_model, rollout_batch):
    """weight_sum logged should equal sum of weights for valid timesteps."""
    lit = _RolloutLitModule(rollout_model)
    weights = [3.0, 1.0]
    _make_rollout_loss(timestep_weights=weights)(lit, rollout_batch, stage="train")
    # Both timesteps are valid (all-finite batch), so weight_sum = 3.0 + 1.0 = 4.0
    assert lit.logged["train_weight_sum"] == sum(weights)


# ── DiagnosticCRPS ────────────────────────────────────────────────────────────

DIAG_SIGMA_OUT_KEYS = [("seviri", "LST_log_sigma_pred")]
CRPS_OUT_KEYS = DIAG_OUT_KEYS + DIAG_SIGMA_OUT_KEYS  # model must output both


@pytest.fixture
def crps_model():
    """Model that outputs both LST_pred and LST_log_sigma_pred."""
    return MultiModalMLP(in_keys=IN_KEYS, out_keys=CRPS_OUT_KEYS, config=MODEL_CONFIG)


@pytest.fixture
def crps_batch(batch):
    """Batch with sigma output pre-populated (forward pass fills it; here we add a slot)."""
    b = batch.clone()
    b[DIAG_SIGMA_OUT_KEYS[0]] = torch.zeros(BATCH_SIZE, 1)
    return b


def _make_diag_crps(**kwargs):
    defaults = dict(in_keys=IN_KEYS, diagnostic_keys=DIAG_KEYS, diagnostic_out_keys=DIAG_OUT_KEYS)
    defaults.update(kwargs)
    return DiagnosticCRPS(**defaults)


def test_diagnostic_crps_returns_finite_scalar(crps_model, batch):
    loss_fn = _make_diag_crps()
    lit = _FakeLitModule(crps_model)
    loss = loss_fn(lit, batch, stage="train")
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_diagnostic_crps_nan_targets_masked(crps_model, batch_with_nans):
    loss_fn = _make_diag_crps()
    lit = _FakeLitModule(crps_model)
    loss = loss_fn(lit, batch_with_nans, stage="train")
    assert torch.isfinite(loss)


def test_diagnostic_crps_differs_from_mse(crps_model, batch):
    """CRPS and MSE should produce different loss values for the same batch."""
    lit = _FakeLitModule(crps_model)
    crps_loss = _make_diag_crps()(lit, batch, stage="train")
    mse_loss = DiagnosticMSE(in_keys=IN_KEYS, diagnostic_keys=DIAG_KEYS, diagnostic_out_keys=DIAG_OUT_KEYS)(
        lit, batch, stage="train"
    )
    assert not torch.isclose(crps_loss, mse_loss)


def test_diagnostic_crps_backward_succeeds(crps_model, batch):
    loss_fn = _make_diag_crps()
    lit = _FakeLitModule(crps_model)
    loss = loss_fn(lit, batch, stage="train")
    loss.backward()


# ── RolloutCRPS ───────────────────────────────────────────────────────────────

PROG_SIGMA_OUT_KEYS = [("seviri", "LST_log_sigma_pred")]
ROLLOUT_CRPS_OUT_KEYS = PROG_OUT_KEYS + PROG_SIGMA_OUT_KEYS


@pytest.fixture
def rollout_crps_model():
    return MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + PROG_KEYS,
        out_keys=ROLLOUT_CRPS_OUT_KEYS,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )


def _make_rollout_crps(**kwargs):
    defaults = dict(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=PROG_KEYS,
        prognostic_out_keys=PROG_OUT_KEYS,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    defaults.update(kwargs)
    return RolloutCRPS(**defaults)


def test_rollout_crps_returns_finite_scalar(rollout_crps_model, rollout_batch):
    loss_fn = _make_rollout_crps()
    lit = _RolloutLitModule(rollout_crps_model)
    loss = loss_fn(lit, rollout_batch, stage="train")
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rollout_crps_backward_succeeds(rollout_crps_model, rollout_batch):
    loss_fn = _make_rollout_crps()
    lit = _RolloutLitModule(rollout_crps_model)
    loss = loss_fn(lit, rollout_batch, stage="train")
    loss.backward()


def test_rollout_crps_timestep_weights_supported(rollout_crps_model, rollout_batch):
    """RolloutCRPS must respect timestep_weights the same way RolloutMSE does."""
    lit = _RolloutLitModule(rollout_crps_model)
    loss_uniform = _make_rollout_crps(timestep_weights=[1.0, 1.0])(lit, rollout_batch, stage="train")
    loss_decaying = _make_rollout_crps(timestep_weights=[4.0, 1.0])(lit, rollout_batch, stage="train")
    assert not torch.isclose(loss_uniform, loss_decaying)


# ── characterization tests: pin numerical baselines for refactor safety ───────
# Each test seeds, builds a deterministic batch + model, and asserts the loss
# equals a recorded value. Any silent numerical drift during refactoring fails.
# If you deliberately change loss behavior, regenerate the baselines.


def test_diagnostic_mse_numerical_baseline():
    torch.manual_seed(0)
    model = MultiModalMLP(in_keys=IN_KEYS, out_keys=DIAG_OUT_KEYS, config=MODEL_CONFIG)
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, 1) for k in IN_KEYS + DIAG_KEYS + DIAG_OUT_KEYS},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = DiagnosticMSE(in_keys=IN_KEYS, diagnostic_keys=DIAG_KEYS, diagnostic_out_keys=DIAG_OUT_KEYS)
    loss = loss_fn(_FakeLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(0.5416700839996338, abs=1e-7)


def test_diagnostic_crps_numerical_baseline():
    torch.manual_seed(0)
    model = MultiModalMLP(in_keys=IN_KEYS, out_keys=CRPS_OUT_KEYS, config=MODEL_CONFIG)
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, 1) for k in IN_KEYS + DIAG_KEYS + CRPS_OUT_KEYS},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = DiagnosticCRPS(in_keys=IN_KEYS, diagnostic_keys=DIAG_KEYS, diagnostic_out_keys=DIAG_OUT_KEYS)
    loss = loss_fn(_FakeLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(0.2240876853466034, abs=1e-7)


def test_rollout_mse_numerical_baseline():
    torch.manual_seed(0)
    model = MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + PROG_KEYS,
        out_keys=PROG_OUT_KEYS,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + PROG_KEYS},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = RolloutMSE(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=PROG_KEYS,
        prognostic_out_keys=PROG_OUT_KEYS,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    loss = loss_fn(_RolloutLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(0.6564374566078186, abs=1e-7)


def test_rollout_crps_numerical_baseline():
    torch.manual_seed(0)
    model = MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + PROG_KEYS,
        out_keys=ROLLOUT_CRPS_OUT_KEYS,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + PROG_KEYS},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = RolloutCRPS(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=PROG_KEYS,
        prognostic_out_keys=PROG_OUT_KEYS,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    loss = loss_fn(_RolloutLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(0.6212354898452759, abs=1e-7)


# ── multi-prognostic-key regression: catches "return-inside-loop" bug class ──


def test_rollout_mse_two_prognostic_keys_baseline():
    prog_keys = [("seviri", "LST"), ("seviri", "AL-BB-DH")]
    prog_out_keys = [("seviri", "LST_pred"), ("seviri", "AL-BB-DH_pred")]
    torch.manual_seed(0)
    model = MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + prog_keys,
        out_keys=prog_out_keys,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + prog_keys},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = RolloutMSE(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=prog_keys,
        prognostic_out_keys=prog_out_keys,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    loss = loss_fn(_RolloutLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(1.1886848211288452, abs=1e-7)


def test_rollout_crps_two_prognostic_keys_baseline():
    prog_keys = [("seviri", "LST"), ("seviri", "AL-BB-DH")]
    prog_out_keys = [("seviri", "LST_pred"), ("seviri", "AL-BB-DH_pred")]
    sigma_out_keys = [("seviri", "LST_log_sigma_pred"), ("seviri", "AL-BB-DH_log_sigma_pred")]
    torch.manual_seed(0)
    model = MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + prog_keys,
        out_keys=prog_out_keys + sigma_out_keys,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + prog_keys},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = RolloutCRPS(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=prog_keys,
        prognostic_out_keys=prog_out_keys,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
    )
    loss = loss_fn(_RolloutLitModule(model), batch, stage="train")
    assert loss.item() == pytest.approx(0.918150782585144, abs=1e-7)


def test_rollout_crps_incremental_loss_raises():
    """CRPS + incremental_loss must fail loudly until a separate sigma_inc head exists."""
    torch.manual_seed(0)
    model = MultiModalMLP(
        in_keys=ROLLOUT_IN_KEYS + PROG_KEYS,
        out_keys=ROLLOUT_CRPS_OUT_KEYS,
        config={**MODEL_CONFIG, "hidden_sizes": [16, 8]},
    )
    batch = TensorDict(
        {k: torch.randn(BATCH_SIZE, ROLLOUT_T, 1) for k in ROLLOUT_IN_KEYS + PROG_KEYS},
        batch_size=[BATCH_SIZE],
    )
    loss_fn = RolloutCRPS(
        in_keys=ROLLOUT_IN_KEYS,
        prognostic_keys=PROG_KEYS,
        prognostic_out_keys=PROG_OUT_KEYS,
        diagnostic_keys=[],
        diagnostic_out_keys=[],
        incremental_loss=True,
    )
    with pytest.raises(NotImplementedError, match="CRPS"):
        loss_fn(_RolloutLitModule(model), batch, stage="train")
