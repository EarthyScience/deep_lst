"""Tests for deep_lst/evaluation/inference.py

Hermetic — no zarr stores, no real data, no GPU required.

Covered:
  _write_init_times
    - scatter into whole init_time slots (time-major row blocks)
    - pixels outside the sparse set stay NaN
    - values land at the right (init_time, lead, row, col); lead_offset shifts leads

  _predict_diagnostic
    - output keys match diagnostic_keys
    - output shape: (N_total_samples, 1)
    - normaliser.denormalise is applied when provided

  _predict_rollout  ← core lookback=0 regression
    - loop runs exactly `horizon` times per batch (NOT horizon+1)
    - output shape is (N_total, horizon) — t=0 is NEVER a forecast target
    - model is called `horizon × n_batches` times in total
    - prognostic state is initialised from batch[:,0,...] (observed t=0)
    - state is updated autoregressively (each step uses previous prediction)
    - NaN in init state (batch[:,0,...]) propagates to all forecast steps
    - all-NaN init returns correct shape (NaN outputs, not crash)
"""

import numpy as np
import torch
from tensordict import TensorDict

from deep_lst.evaluation.inference import Forecast

# ── shared constants ──────────────────────────────────────────────────────────

B = 4  # batch size
HORIZON = 3  # forecast horizon steps
LOOKBACK = 0  # must remain 0 after the lookback fix
T = HORIZON + 1  # total time steps in a batch window (lookback=0 → 4)

IN_KEYS = [("era5", "t2m"), ("era5", "u10")]
PROG_KEYS = [("seviri", "LST")]
PROG_OUT_KEYS = [("seviri", "LST_pred")]
DIAG_KEYS = [("seviri", "FVC")]
DIAG_OUT_KEYS = [("seviri", "FVC_pred")]

ROLLOUT_TD_KEYS = {
    "in_keys": IN_KEYS,
    "prognostic_keys": PROG_KEYS,
    "prognostic_out_keys": PROG_OUT_KEYS,
    "diagnostic_keys": [],
    "diagnostic_out_keys": [],
}

DIAG_TD_KEYS = {
    "in_keys": IN_KEYS,
    "prognostic_keys": [],
    "prognostic_out_keys": [],
    "diagnostic_keys": DIAG_KEYS,
    "diagnostic_out_keys": DIAG_OUT_KEYS,
}

# Diagnostic batch shape: (B, 1, 1, 1) — _predict_diagnostic squeezes (-3, -2, -1).
SPATIAL = (1, 1, 1)

# Packed-tensor layout (post-collator) consumed by _predict_rollout:
#   y_prog    (B, T, H, W, nP+nD)  prognostics first, then diagnostics
#   X_dyn     (B, T, H, W, nF_all) step forcings
#   X_dyn_ctx (B, T, H, W, nF_ctx) lookback-window forcings (empty slice at lookback=0)
# Single pixel (H=W=1); the rollout squeezes H,W to yield (B, horizon) per variable.
N_PROG = 1  # prognostic channels in y_prog (matches PROG_KEYS)
N_DIAG = 0  # diagnostic channels (rollout tests use none)
N_TARGET = N_PROG + N_DIAG
N_INPUT = 2  # X_dyn channels (forcings the mock models ignore)
N_CTX = 1  # X_dyn_ctx channels (unused at lookback=0)
GRID = (1, 1)  # H, W


def _rollout_batch(b=B, t=T):
    """Packed-tensor TensorDict (post-collator format) for the rollout path.

    Channel 0 of y_prog is the single prognostic var (PROG_KEYS[0]); tests set the
    init state via y_prog[:, 0, ..., 0] and read targets from y_prog[:, 1:, ..., 0].
    """
    data = {
        "y_prog": torch.randn(b, t, *GRID, N_TARGET),
        "X_dyn": torch.randn(b, t, *GRID, N_INPUT),
        "X_dyn_ctx": torch.randn(b, t, *GRID, N_CTX),
    }
    return TensorDict(data, batch_size=[b])


def _diag_batch(b=B, finite=True):
    """TensorDict with shape (b, *SPATIAL) for diagnostic keys (no time dim)."""
    data = {}
    for k in IN_KEYS:
        data[k] = torch.randn(b, *SPATIAL)
    data[DIAG_KEYS[0]] = torch.randn(b, *SPATIAL)
    data[DIAG_OUT_KEYS[0]] = torch.zeros(b, *SPATIAL)
    return TensorDict(data, batch_size=[b])


# Placeholder config carried by the mock models; neither _predict_rollout nor
# _predict_diagnostic reads model.config in the current packed-tensor API.
_EMPTY_TRAIN_CONFIG = {"train_dataset": {}, "model": {}}


class _ZeroIncrementModel:
    """Rollout model: returns a zero increment (and zero diagnostics) every step.

    Signature matches MultiModalMLP.forward(state_window, X_dyn_window, step_X_dyn).
    """

    def __init__(self):
        self.model = self
        self.call_count = 0

    def __call__(self, state_window, X_dyn_window, step_X_dyn):
        self.call_count += 1
        n, h, w = step_X_dyn.shape[:3]
        return torch.zeros(n, h, w, N_TARGET)


class _ConstantIncrementModel:
    """Rollout model: returns a +1.0 increment for every prognostic channel."""

    def __init__(self):
        self.model = self
        self.call_count = 0

    def __call__(self, state_window, X_dyn_window, step_X_dyn):
        self.call_count += 1
        n, h, w = step_X_dyn.shape[:3]
        out = torch.zeros(n, h, w, N_TARGET)
        out[..., :N_PROG] = 1.0
        return out


class _ConstantDiagModel:
    """Diagnostic model: always outputs a fixed value for the diagnostic key."""

    def __init__(self, value=2.5):
        self.model = self
        self.config = _EMPTY_TRAIN_CONFIG
        self.value = value

    def __call__(self, td):
        result = td.clone()
        result[DIAG_OUT_KEYS[0]] = torch.full_like(td[DIAG_KEYS[0]], self.value)
        return result


class _MockNormaliser:
    """Identity normaliser that records denormalise calls."""

    def __init__(self):
        self.calls = 0

    def denormalise(self, arr, var):
        self.calls += 1
        return arr  # identity


def _collect(gen):
    """Drain a predict generator and concatenate per-batch dicts (the old return shape)."""
    batches = list(gen)
    preds = {v: np.concatenate([p[v] for p, _ in batches], axis=0) for v in batches[0][0]}
    obs = {v: np.concatenate([o[v] for _, o in batches], axis=0) for v in batches[0][1]}
    return preds, obs


def _run_rollout(*args, **kwargs):
    return _collect(Forecast._predict_rollout(*args, **kwargs))


def _run_diagnostic(*args, **kwargs):
    return _collect(Forecast._predict_diagnostic(*args, **kwargs))


# ── _write_init_times ──────────────────────────────────────────────────────────


def _empty_store(n_time, t_out, grid_shape):
    """Numpy stand-ins for the zarr pred/obs arrays (same slice-assign interface)."""
    pa = np.full((n_time, t_out, *grid_shape), np.nan, dtype=np.float32)
    oa = np.full((n_time, t_out, *grid_shape), np.nan, dtype=np.float32)
    return pa, oa


def test_write_init_times_scatter_and_nan():
    """Values land at their (row, col); grid cells outside the pixel set stay NaN."""
    n_sp, t = 2, 2
    rows, cols = np.array([0, 1]), np.array([0, 1])
    pa, oa = _empty_store(n_time=1, t_out=t, grid_shape=(3, 3))
    preds = {"LST": np.ones((n_sp, t), dtype=np.float32)}
    obs = {"LST": np.ones((n_sp, t), dtype=np.float32)}

    Forecast._write_init_times({"LST": (pa, oa)}, preds, obs, 0, n_sp, 0, rows, cols, (3, 3), sparse=False)

    assert pa[0, 0, 0, 0] == 1.0
    assert pa[0, 0, 1, 1] == 1.0
    assert np.isnan(pa[0, 0, 0, 1]), "Pixel (0,1) was not in sparse set — should stay NaN"
    assert np.isnan(pa[0, 0, 1, 0]), "Pixel (1,0) was not in sparse set — should stay NaN"


def test_write_init_times_values_preserved():
    """Time-major rows: row j of an init_time block is pixel j, leads along axis 1."""
    rows, cols = np.array([0, 0]), np.array([0, 1])
    pa, oa = _empty_store(n_time=1, t_out=2, grid_shape=(1, 2))
    preds = {"LST": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)}
    obs = {"LST": preds["LST"].copy()}

    Forecast._write_init_times({"LST": (pa, oa)}, preds, obs, 0, 2, 0, rows, cols, (1, 2), sparse=False)

    assert pa[0, 0, 0, 0] == 10.0  # pix0, lead0
    assert pa[0, 1, 0, 0] == 20.0  # pix0, lead1
    assert pa[0, 0, 0, 1] == 30.0  # pix1, lead0
    assert pa[0, 1, 0, 1] == 40.0  # pix1, lead1


def test_write_init_times_multiple_slots_and_lead_offset():
    """k init_times per call land at ti_start..ti_start+k-1; lead_offset=1 keeps lead 0 NaN."""
    n_sp = 2
    rows, cols = np.array([0, 0]), np.array([0, 1])
    pa, oa = _empty_store(n_time=4, t_out=2, grid_shape=(1, 2))
    # Two init_time blocks of n_sp rows each, one lead: block 0 → 1.0, block 1 → 2.0.
    preds = {"LST": np.array([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)}
    obs = {"LST": preds["LST"].copy()}

    Forecast._write_init_times({"LST": (pa, oa)}, preds, obs, 1, n_sp, 1, rows, cols, (1, 2), sparse=False)

    assert np.isnan(pa[0]).all(), "init_time 0 was not written — must stay NaN"
    assert np.isnan(pa[1, 0]).all(), "lead 0 must stay NaN with lead_offset=1"
    assert (pa[1, 1] == 1.0).all()
    assert (pa[2, 1] == 2.0).all()
    assert np.isnan(pa[3]).all(), "init_time 3 was not written — must stay NaN"


def test_write_init_times_sparse_transpose():
    """Sparse layout is (init_time, lead, pixel): row blocks transpose to (lead, pixel)."""
    n_sp = 3
    pa = np.full((1, 2, n_sp), np.nan, dtype=np.float32)
    oa = pa.copy()
    preds = {"LST": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)}  # (pixel, lead)
    obs = {"LST": preds["LST"].copy()}

    Forecast._write_init_times({"LST": (pa, oa)}, preds, obs, 0, n_sp, 0, None, None, None, sparse=True)

    np.testing.assert_array_equal(pa[0, 0], [1.0, 3.0, 5.0])  # lead 0 across pixels
    np.testing.assert_array_equal(pa[0, 1], [2.0, 4.0, 6.0])  # lead 1 across pixels


# ── _predict_diagnostic ───────────────────────────────────────────────────────


def test_predict_diagnostic_output_keys():
    loader = [_diag_batch() for _ in range(2)]
    preds, obs = _run_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert var in preds
    assert var in obs


def test_predict_diagnostic_output_shape():
    n_batches = 3
    loader = [_diag_batch() for _ in range(n_batches)]
    preds, obs = _run_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert preds[var].shape[0] == n_batches * B


def test_predict_diagnostic_constant_model_output():
    """Model output of constant 2.5 should be reflected in predictions after denorm."""
    loader = [_diag_batch() for _ in range(2)]
    preds, _ = _run_diagnostic(_ConstantDiagModel(value=2.5), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert np.allclose(preds[var], 2.5), "Constant model output should be preserved exactly"


def test_predict_diagnostic_normaliser_called():
    """denormalise() must be invoked once per variable per batch."""
    n_batches = 3
    loader = [_diag_batch() for _ in range(n_batches)]
    norm = _MockNormaliser()
    _run_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS, normaliser=norm)
    # denormalise called for preds AND obs for 1 variable × n_batches × 2 sides
    assert norm.calls == n_batches * 2


# ── _predict_rollout: lookback=0 regression ───────────────────────────────────


def test_rollout_output_shape_is_N_by_horizon():
    """
    REGRESSION: output shape must be (N_total, HORIZON), not (N_total, HORIZON+1).

    Before the lookback=0 fix, the loop was range(lookback, horizon+1). With
    lookback=0 that would iterate t=0,1,2,3 → 4 steps → shape (N, 4).
    After the fix, the loop is range(1, horizon+1) → t=1,2,3 → 3 steps → shape (N, 3).
    """
    n_batches = 2
    loader = [_rollout_batch() for _ in range(n_batches)]
    model = _ZeroIncrementModel()

    preds, obs = _run_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    expected_shape = (n_batches * B, HORIZON)
    assert preds[var].shape == expected_shape, (
        f"Shape is {preds[var].shape}, expected {expected_shape}. "
        f"If shape[1]=={HORIZON + 1} the loop starts at t=0 (lookback bug)."
    )
    assert obs[var].shape == expected_shape


def test_rollout_model_called_horizon_times_per_batch():
    """Model must be called exactly horizon × n_batches times total."""
    n_batches = 3
    loader = [_rollout_batch() for _ in range(n_batches)]
    model = _ZeroIncrementModel()

    _run_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    expected_calls = n_batches * HORIZON
    assert model.call_count == expected_calls, (
        f"Model called {model.call_count} times, expected {expected_calls}. "
        f"If called {n_batches * (HORIZON + 1)} times the loop includes t=0."
    )


def test_rollout_init_state_is_t0_observed():
    """
    The rollout must anchor on the observed t=0 state (y_prog[:, 0]).

    At lookback=0 the model receives no state window — the autoregressive carry is
    the rollout's `prior = state_rollout[:, t-1]`. With a zero-increment model the
    state never changes, so every lead-time prediction equals the t=0 observation,
    proving the first prior was the observed init and not a later target.
    """
    b = _rollout_batch(b=1)  # single sample for easy comparison
    b["y_prog"][:, 0, ..., 0] = 7.0  # observed init state
    b["y_prog"][:, 1:, ..., 0] = -1.0  # later targets must not leak into preds
    loader = [b]

    preds, _ = _run_rollout(
        _ZeroIncrementModel(),
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    np.testing.assert_allclose(
        preds[var],
        7.0,
        err_msg="Predictions are not anchored on the observed t=0 state.",
    )


def test_rollout_t0_not_a_forecast_target():
    """
    The forecast output at each step corresponds to targets at t=1,2,...,horizon.
    Target at t=0 (the init observation itself) must never appear in predictions.

    Strategy: set batch[:,0,...] = 99.0 (a sentinel), batch[:,t>0,...] = normal values.
    With zero increment, preds == init_state (99.0). If t=0 were a target, obs
    at index 0 in the output would be 99.0. But targets are batch[:,1,...],
    batch[:,2,...], batch[:,3,...] — none of which are 99.0.
    """
    b = _rollout_batch()
    b["y_prog"][:, 0, ..., 0] = 99.0  # sentinel at t=0 (init, not a target)
    b["y_prog"][:, 1:, ..., 0] = 0.0  # targets t=1..horizon = 0.0
    loader = [b]
    model = _ZeroIncrementModel()

    preds, obs = _run_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    # obs must come from y_prog[:,1], y_prog[:,2], y_prog[:,3] — all 0.0
    assert not np.any(obs[var] == 99.0), (
        "obs contains the t=0 sentinel value (99.0) — t=0 is being used as a "
        "forecast target, which violates the lookback=0 convention."
    )
    assert np.allclose(obs[var], 0.0), f"All obs targets should be 0.0, got {obs[var]}"


def test_rollout_autoregressive_state_update():
    """
    With +1 increment model, pred at step k should be init + k.
    This verifies that prog is updated after each step (autoregressive rollout).
    """
    init_val = 5.0
    b = _rollout_batch(b=1)
    b["y_prog"][:, :, ..., 0] = init_val  # all timesteps 5.0; only t=0 used as init
    loader = [b]
    model = _ConstantIncrementModel()

    preds, _ = _run_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    # step 1: init(5) + 1 = 6; step 2: 6+1=7; step 3: 7+1=8
    expected = np.array([[init_val + k for k in range(1, HORIZON + 1)]])  # (1, horizon)
    np.testing.assert_allclose(preds[var], expected, rtol=1e-5)


def test_rollout_nan_init_state_propagates():
    """All-NaN init state (t=0) should produce NaN outputs for all lead times."""
    b = _rollout_batch()
    b["y_prog"][:, 0, ..., 0] = float("nan")  # corrupt init
    loader = [b]
    model = _ZeroIncrementModel()

    preds, _ = _run_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    # With NaN init and any increment, outputs will be NaN
    assert np.all(np.isnan(preds[var])), "Expected all-NaN predictions when init state is NaN."


def test_rollout_nan_init_correct_output_shape():
    """Even with NaN init, output shape must be (N_total, horizon)."""
    b = _rollout_batch()
    b["y_prog"][:, 0, ..., 0] = float("nan")
    loader = [b]
    preds, obs = _run_rollout(
        _ZeroIncrementModel(),
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )
    var = PROG_KEYS[0][1]
    assert preds[var].shape == (B, HORIZON)


def test_rollout_normaliser_called_per_batch():
    """denormalise is applied once per (var, side) per batch — on the stacked
    (B, horizon) tensor, not per rollout step."""
    n_batches = 2
    loader = [_rollout_batch() for _ in range(n_batches)]
    norm = _MockNormaliser()

    _run_rollout(
        _ZeroIncrementModel(),
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=norm,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    # 1 prognostic var × 2 sides (pred + obs) × n_batches
    expected_calls = 1 * 2 * n_batches
    assert norm.calls == expected_calls
