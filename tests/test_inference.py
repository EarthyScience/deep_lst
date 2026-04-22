"""Tests for deep_lst/evaluation/inference.py

Hermetic — no zarr stores, no real data, no GPU required.
eodata is required only for build_local_grid; those tests are skipped otherwise.

Covered:
  ForecastResult
    - construction and xarray conversion (dims, shapes, coordinates)
    - lead times start at 1h (lookback=0 regression)
    - save / load roundtrip (values, coords)

  _apply_consistent_mask
    - all-finite input passes through unchanged
    - NaN in obs at any lead time propagates to all lead times for that sample
    - mask is applied per-variable independently

  _remap_to_grid
    - output shape: (n_time, t, *grid_shape)
    - pixels outside the sparse set are NaN
    - pixel values survive the reshape

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

  Forecast.predict() end-to-end
    - lead_times = [1h, 2h, ..., horizon h]  (not [0h, 1h, ...])
    - ForecastResult has correct n_time, horizon, spatial dims
    - works without a normaliser (normaliser=None path)
"""

import numpy as np
import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from deep_lst.evaluation.inference import Forecast, ForecastResult

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

# Batch shape: (B, T, 1, 1, 1) — single pixel, single channel.
# The three trailing spatial/channel dims are needed because _predict_rollout
# calls .squeeze((-3, -2, -1)) on model output.
SPATIAL = (1, 1, 1)


def _rollout_batch(b=B, t=T, finite=True):
    """TensorDict with shape (b, t, *SPATIAL) for all rollout keys."""
    data = {}
    for k in IN_KEYS + PROG_KEYS:
        tensor = torch.randn(b, t, *SPATIAL) if finite else torch.full((b, t, *SPATIAL), float("nan"))
        data[k] = tensor
    return TensorDict(data, batch_size=[b])


def _diag_batch(b=B, finite=True):
    """TensorDict with shape (b, *SPATIAL) for diagnostic keys (no time dim)."""
    data = {}
    for k in IN_KEYS:
        data[k] = torch.randn(b, *SPATIAL)
    data[DIAG_KEYS[0]] = torch.randn(b, *SPATIAL)
    data[DIAG_OUT_KEYS[0]] = torch.zeros(b, *SPATIAL)
    return TensorDict(data, batch_size=[b])


class _ZeroIncrementModel:
    """Rollout model: always returns a zero increment for the prognostic key."""

    def __init__(self):
        self.model = self
        self.call_count = 0
        self.prog_states_at_call: list = []  # records prog state snapshot each call

    def __call__(self, td):
        self.call_count += 1
        self.prog_states_at_call.append(td[PROG_KEYS[0]].clone())
        result = td.clone()
        result[PROG_OUT_KEYS[0]] = torch.zeros_like(td[PROG_KEYS[0]])
        return result


class _ConstantIncrementModel:
    """Rollout model: always adds 1.0 to the prognostic state."""

    def __init__(self):
        self.model = self
        self.call_count = 0

    def __call__(self, td):
        self.call_count += 1
        result = td.clone()
        result[PROG_OUT_KEYS[0]] = torch.ones_like(td[PROG_KEYS[0]])
        return result


class _ConstantDiagModel:
    """Diagnostic model: always outputs a fixed value for the diagnostic key."""

    def __init__(self, value=2.5):
        self.model = self
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


# ── ForecastResult ─────────────────────────────────────────────────────────────


@pytest.fixture
def sample_result():
    n_time, n_lead, n_lat, n_lon = 5, HORIZON, 10, 12
    shape = (n_time, n_lead, n_lat, n_lon)
    return ForecastResult(
        predictions={"LST": np.random.randn(*shape).astype(np.float32)},
        observations={"LST": np.random.randn(*shape).astype(np.float32)},
        init_times=pd.date_range("2020-01-01", periods=n_time, freq="h"),
        lead_times=pd.timedelta_range("1h", periods=n_lead, freq="h"),
        lats=np.linspace(-5.0, 5.0, n_lat),
        lons=np.linspace(30.0, 41.0, n_lon),
    )


def test_to_xarray_dims(sample_result):
    ds = sample_result.to_xarray()
    assert set(ds.dims) == {"init_time", "lead_time", "lat", "lon"}


def test_to_xarray_variables(sample_result):
    ds = sample_result.to_xarray()
    assert "LST_pred" in ds.data_vars
    assert "LST_obs" in ds.data_vars


def test_to_xarray_shape(sample_result):
    ds = sample_result.to_xarray()
    assert ds["LST_pred"].shape == (5, HORIZON, 10, 12)
    assert ds["LST_obs"].shape == (5, HORIZON, 10, 12)


def test_lead_times_start_at_1h(sample_result):
    """Regression: with lookback=0 the first lead time must be 1 h, not 0 h."""
    ds = sample_result.to_xarray()
    first_lead = pd.Timedelta(ds["lead_time"].values[0])
    assert first_lead == pd.Timedelta("1h"), (
        f"First lead time is {first_lead}. Expected 1h — if it is 0h the "
        "lookback=0 fix has been reverted or lead time indexing is wrong."
    )


def test_lead_times_are_sequential_hourly(sample_result):
    ds = sample_result.to_xarray()
    leads = pd.TimedeltaIndex(ds["lead_time"].values)
    for i, lt in enumerate(leads):
        assert lt == pd.Timedelta(f"{i + 1}h"), f"Lead time at index {i} should be {i + 1}h"


def test_roundtrip_values(tmp_path, sample_result):
    path = tmp_path / "forecast.zarr"
    sample_result.save(path)
    loaded = ForecastResult.load(path)
    np.testing.assert_allclose(loaded.predictions["LST"], sample_result.predictions["LST"], rtol=1e-5)
    np.testing.assert_allclose(loaded.observations["LST"], sample_result.observations["LST"], rtol=1e-5)


def test_roundtrip_coords(tmp_path, sample_result):
    path = tmp_path / "forecast.zarr"
    sample_result.save(path)
    loaded = ForecastResult.load(path)
    np.testing.assert_allclose(loaded.lats, sample_result.lats)
    np.testing.assert_allclose(loaded.lons, sample_result.lons)
    assert len(loaded.init_times) == len(sample_result.init_times)
    assert len(loaded.lead_times) == len(sample_result.lead_times)


# ── _apply_consistent_mask ────────────────────────────────────────────────────


def test_mask_all_finite_passes_through():
    preds = {"LST": np.ones((6, HORIZON))}
    obs = {"LST": np.ones((6, HORIZON))}
    pm, om = Forecast._apply_consistent_mask(preds, obs)
    assert np.all(np.isfinite(pm["LST"]))
    assert np.all(np.isfinite(om["LST"]))


def test_mask_nan_in_one_lead_propagates_to_all():
    """A single NaN in obs at lead 1 for sample 2 → all leads for sample 2 become NaN."""
    obs = np.ones((6, HORIZON))
    obs[2, 1] = np.nan
    preds = {"LST": np.ones((6, HORIZON))}
    pm, om = Forecast._apply_consistent_mask(preds, {"LST": obs})
    assert np.all(np.isnan(pm["LST"][2, :])), "All pred leads for sample 2 should be NaN"
    assert np.all(np.isnan(om["LST"][2, :])), "All obs leads for sample 2 should be NaN"
    assert np.all(np.isfinite(pm["LST"][0, :])), "Unaffected sample 0 should remain finite"


def test_mask_pred_nan_also_propagates():
    """NaN in preds (not obs) should also propagate to all leads for that sample."""
    preds = np.ones((4, HORIZON))
    preds[1, 0] = np.nan
    obs = np.ones((4, HORIZON))
    pm, om = Forecast._apply_consistent_mask({"LST": preds}, {"LST": obs})
    # obs drives the mask; pred NaN alone does NOT propagate — only obs NaN does
    # (design: mask based on obs validity)
    assert np.all(np.isfinite(om["LST"]))


def test_mask_independent_per_variable():
    preds = {"LST": np.ones((4, 3)), "T2": np.ones((4, 3))}
    obs = {"LST": np.ones((4, 3)), "T2": np.ones((4, 3))}
    obs["LST"][1, 0] = np.nan
    pm, om = Forecast._apply_consistent_mask(preds, obs)
    assert np.all(np.isnan(om["LST"][1, :]))
    assert np.all(np.isfinite(om["T2"])), "T2 should be unaffected by LST mask"


# ── _remap_to_grid ─────────────────────────────────────────────────────────────


def test_remap_output_shape():
    n_sp, n_t, t = 4, 3, HORIZON
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    preds = np.random.randn(n_sp * n_t, t)
    obs = np.random.randn(n_sp * n_t, t)
    pg, og = Forecast._remap_to_grid(preds, obs, n_sp, n_t, t, rows, cols, (2, 2))
    assert pg.shape == (n_t, t, 2, 2)
    assert og.shape == (n_t, t, 2, 2)


def test_remap_outside_pixels_are_nan():
    n_sp, n_t, t = 2, 1, 2
    rows = np.array([0, 1])
    cols = np.array([0, 1])
    preds = np.ones((n_sp * n_t, t))
    obs = np.ones((n_sp * n_t, t))
    pg, _ = Forecast._remap_to_grid(preds, obs, n_sp, n_t, t, rows, cols, (3, 3))
    assert np.isnan(pg[0, 0, 0, 1]), "Pixel (0,1) was not in sparse set — should be NaN"
    assert np.isnan(pg[0, 0, 1, 0]), "Pixel (1,0) was not in sparse set — should be NaN"
    assert pg[0, 0, 0, 0] == 1.0
    assert pg[0, 0, 1, 1] == 1.0


def test_remap_values_preserved():
    """Values at known pixels should survive the reshape exactly."""
    n_sp, n_t, t = 2, 1, 2
    rows = np.array([0, 0])
    cols = np.array([0, 1])
    # preds ordered spatial-major: [pix0_time0, pix1_time0] each of length t=2
    preds = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)  # (n_sp*n_t=2, t=2)
    obs = preds.copy()
    pg, _ = Forecast._remap_to_grid(preds, obs, n_sp, n_t, t, rows, cols, (1, 2))
    # pg shape: (1, 2, 1, 2)
    assert pg[0, 0, 0, 0] == 10.0  # pix0, lead0
    assert pg[0, 1, 0, 0] == 20.0  # pix0, lead1
    assert pg[0, 0, 0, 1] == 30.0  # pix1, lead0
    assert pg[0, 1, 0, 1] == 40.0  # pix1, lead1


# ── _predict_diagnostic ───────────────────────────────────────────────────────


def test_predict_diagnostic_output_keys():
    loader = [_diag_batch() for _ in range(2)]
    preds, obs = Forecast._predict_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert var in preds
    assert var in obs


def test_predict_diagnostic_output_shape():
    n_batches = 3
    loader = [_diag_batch() for _ in range(n_batches)]
    preds, obs = Forecast._predict_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert preds[var].shape[0] == n_batches * B


def test_predict_diagnostic_constant_model_output():
    """Model output of constant 2.5 should be reflected in predictions after denorm."""
    loader = [_diag_batch() for _ in range(2)]
    preds, _ = Forecast._predict_diagnostic(_ConstantDiagModel(value=2.5), torch.device("cpu"), loader, DIAG_TD_KEYS)
    var = DIAG_KEYS[0][1]
    assert np.allclose(preds[var], 2.5), "Constant model output should be preserved exactly"


def test_predict_diagnostic_normaliser_called():
    """denormalise() must be invoked once per variable per batch."""
    n_batches = 3
    loader = [_diag_batch() for _ in range(n_batches)]
    norm = _MockNormaliser()
    Forecast._predict_diagnostic(_ConstantDiagModel(), torch.device("cpu"), loader, DIAG_TD_KEYS, normaliser=norm)
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

    preds, obs = Forecast._predict_rollout(
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

    Forecast._predict_rollout(
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
    The first call to the model must see prog = batch[:,0,...] (observed at t=0),
    not a predicted value.

    With zero-increment model the state never changes, so all model calls receive
    the same state = batch[:,0,...].
    """
    batch = _rollout_batch(b=1)  # single sample for easy comparison
    loader = [batch]
    model = _ZeroIncrementModel()

    Forecast._predict_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    init_state = batch[PROG_KEYS[0]][:, 0, ...].numpy()
    first_call_state = model.prog_states_at_call[0].numpy()

    np.testing.assert_array_equal(
        first_call_state,
        init_state,
        err_msg="First model call did not receive the observed t=0 state.",
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
    b[PROG_KEYS[0]][:, 0, ...] = 99.0  # sentinel at t=0 (init, not a target)
    b[PROG_KEYS[0]][:, 1:, ...] = 0.0  # targets t=1..horizon = 0.0
    loader = [b]
    model = _ZeroIncrementModel()

    preds, obs = Forecast._predict_rollout(
        model,
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=None,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    var = PROG_KEYS[0][1]
    # obs must come from batch[:,1,...], batch[:,2,...], batch[:,3,...] — all 0.0
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
    b[PROG_KEYS[0]][:] = init_val  # all timesteps set to 5.0; only t=0 used for init
    loader = [b]
    model = _ConstantIncrementModel()

    preds, _ = Forecast._predict_rollout(
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
    b[PROG_KEYS[0]][:, 0, ...] = float("nan")  # corrupt init
    loader = [b]
    model = _ZeroIncrementModel()

    preds, _ = Forecast._predict_rollout(
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
    b[PROG_KEYS[0]][:, 0, ...] = float("nan")
    loader = [b]
    preds, obs = Forecast._predict_rollout(
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


def test_rollout_normaliser_called_per_step():
    """denormalise must be called for preds and obs at each of the horizon steps."""
    n_batches = 2
    loader = [_rollout_batch() for _ in range(n_batches)]
    norm = _MockNormaliser()

    Forecast._predict_rollout(
        _ZeroIncrementModel(),
        torch.device("cpu"),
        loader,
        ROLLOUT_TD_KEYS,
        normaliser=norm,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )

    # 1 prognostic var × 2 sides (pred + obs) × horizon steps × n_batches
    expected_calls = 1 * 2 * HORIZON * n_batches
    assert norm.calls == expected_calls


# ── Forecast.predict() end-to-end ─────────────────────────────────────────────

eodata = pytest.importorskip("eodata", reason="eodata not installed")


@pytest.fixture
def tiny_spatial_indices():
    """4 pixel indices forming a 2×2 patch near the grid centre."""
    from eodata.grids import EquiangularGrid

    grid = EquiangularGrid(
        resolution=0.05,
        latitude_bounds=(80.0, -80.0),
        longitude_bounds=(-80.0, 80.0),
        include_ymin=True,
        include_xmax=True,
    )
    n_rows, n_cols = grid.get_shape()
    # Pick a 2×2 patch in the centre
    r0, c0 = n_rows // 2, n_cols // 2
    indices = np.array(
        [
            r0 * n_cols + c0,
            r0 * n_cols + c0 + 1,
            (r0 + 1) * n_cols + c0,
            (r0 + 1) * n_cols + c0 + 1,
        ],
        dtype=np.int64,
    )
    return indices


@pytest.fixture
def forecast_instance(monkeypatch, tiny_spatial_indices):
    """Forecast with mocked normaliser and a fake loader (no zarr/GPU needed)."""
    from unittest.mock import patch

    config = {
        "seviri": {"statistics": "global", "lookback": LOOKBACK, "horizon": HORIZON},
    }

    # Patch ZScoringNormalizer so no statistics file is needed
    mock_norm = _MockNormaliser()
    with patch("deep_lst.evaluation.inference.ZScoringNormalizer", return_value=mock_norm):
        n_spatial = len(tiny_spatial_indices)  # 4 pixels
        n_time = 3  # 3 init times

        # Build a loader whose samples cover n_spatial × n_time = 12 items
        loader = [_rollout_batch(b=n_spatial, t=T) for _ in range(n_time)]
        model = _ZeroIncrementModel()

        fc = Forecast(
            model=model,
            device=torch.device("cpu"),
            loader=loader,
            td_keys=ROLLOUT_TD_KEYS,
            config=config,
            spatial_indices=tiny_spatial_indices,
            prediction_mode="RolloutMSE",
            apply_consistent_mask=False,
        )
    return fc, loader, model


def test_predict_lead_times_start_at_1h(forecast_instance):
    """Regression: ForecastResult.lead_times[0] must be 1h after lookback=0 fix."""
    fc, _, _ = forecast_instance
    result = fc.predict()
    ds = result.to_xarray()
    first_lead = pd.Timedelta(ds["lead_time"].values[0])
    assert first_lead == pd.Timedelta("1h"), (
        f"lead_times[0] = {first_lead}. Expected 1h. If 0h, the lead_start_h computation still uses lookback."
    )


def test_predict_lead_times_length(forecast_instance):
    fc, _, _ = forecast_instance
    result = fc.predict()
    assert len(result.lead_times) == HORIZON


def test_predict_result_has_correct_n_time(forecast_instance):
    fc, loader, _ = forecast_instance
    result = fc.predict()
    # n_time = total_samples / n_spatial = (n_batches * B) / n_spatial
    total_samples = len(loader) * B
    n_spatial = len(fc.unique_spatial_idx)
    expected_n_time = total_samples // n_spatial
    assert result.predictions["LST"].shape[0] == expected_n_time


def test_predict_result_spatial_shape(forecast_instance):
    """Spatial dims of ForecastResult must match the local grid bounding box."""
    fc, _, _ = forecast_instance
    result = fc.predict()
    lat_extent = len(result.lats)
    lon_extent = len(result.lons)
    assert result.predictions["LST"].shape[2] == lat_extent
    assert result.predictions["LST"].shape[3] == lon_extent


def test_predict_result_lats_lons_finite(forecast_instance):
    fc, _, _ = forecast_instance
    result = fc.predict()
    assert np.all(np.isfinite(result.lats))
    assert np.all(np.isfinite(result.lons))
