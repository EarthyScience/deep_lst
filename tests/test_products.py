import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.conditioning import time_of_day_mask
from deep_lst.evaluation.metrics import (
    compute_pooled_by_day,
    compute_pooled_scores,
    compute_rmse_sufficient_stats,
    compute_scores,
    daily_rmse_by_valid_day,
    skill_score,
)
from deep_lst.evaluation.products import align_baseline, compute_products

RAW = ["rmse", "mae", "bias", "nse"]


def _synthetic_cube(seed=7):
    rng = np.random.default_rng(seed)
    init_times = pd.date_range("2024-01-01", periods=8, freq="6h")
    lead_times = pd.timedelta_range("1h", periods=48, freq="1h")
    shape = (8, 48, 3, 2)
    pred_values = rng.normal(300.0, 5.0, size=shape)
    obs_values = pred_values + rng.normal(0.0, 2.0, size=shape)
    obs_values[rng.random(shape) < 0.15] = np.nan
    pred_values[rng.random(shape) < 0.03] = np.nan
    coords = {"init_time": init_times, "lead_time": lead_times, "lat": [10.0, 10.05, 10.1], "lon": [0.0, 30.0]}
    dims = ("init_time", "lead_time", "lat", "lon")
    return (
        xr.DataArray(pred_values, dims=dims, coords=coords),
        xr.DataArray(obs_values, dims=dims, coords=coords),
    )


def _baseline_cube(obs, seed=11):
    rng = np.random.default_rng(seed)
    return obs + rng.normal(0.0, 3.0, size=obs.shape)


def test_scores_by_lead_matches_compute_pooled_scores():
    pred, obs = _synthetic_cube()
    out = compute_products(pred, obs, ["scores_by_lead"])["scores_by_lead"]
    old = compute_pooled_scores(pred, obs, RAW)
    xr.testing.assert_allclose(out[RAW], old, rtol=1e-5)


def test_scores_by_forecast_day_matches_and_skill():
    pred, obs = _synthetic_cube()
    baseline = _baseline_cube(obs)
    out = compute_products(pred, obs, ["scores_by_forecast_day"], baselines={"pers": baseline})
    product = out["scores_by_forecast_day"]

    old = compute_pooled_by_day(pred, obs, RAW)
    xr.testing.assert_allclose(product[RAW], old, rtol=1e-5)
    assert list(product["day_label"].values) == list(old["day_label"].values)

    old_baseline = compute_pooled_by_day(baseline, obs, RAW)
    expected_skill = skill_score(old["rmse"], old_baseline["rmse"])
    np.testing.assert_allclose(product["rmse_skill_pers"].values, expected_skill.values, rtol=1e-5)


def test_valid_day_product_matches_daily_rmse():
    pred, obs = _synthetic_cube()
    out = compute_products(pred, obs, ["rmse_by_valid_day_lead1_24"])["rmse_by_valid_day_lead1_24"]

    old_daily, _ = daily_rmse_by_valid_day(compute_rmse_sufficient_stats(pred, obs), np.arange(24))
    np.testing.assert_allclose(out["rmse"].values, old_daily.values, rtol=1e-5)
    assert list(out["month"].values) == list(old_daily["month"].values)


def test_anomaly_products_match_registry():
    pred, obs = _synthetic_cube()
    clim = obs * 0 + 300.0  # any shared reference works for the identity
    out = compute_products(pred, obs, ["anomaly_by_init_lead"], clim=clim)["anomaly_by_init_lead"]

    old = compute_scores(pred - clim, obs - clim, ["acc", "activity"], "region_time")
    xr.testing.assert_allclose(out, old, rtol=1e-5, atol=1e-6)  # atol: acc crosses zero


def test_map_product_first_bucket_matches_manual():
    pred, obs = _synthetic_cube()
    out = compute_products(pred, obs, ["rmse_map_by_bucket_tod"])["rmse_map_by_bucket_tod"]

    day_mask = time_of_day_mask(pred, "day")
    lead_hours = (pred["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    in_day1 = pred["lead_time"].values[(lead_hours >= 1) & (lead_hours <= 24)]
    err2 = ((pred - obs) ** 2).where(day_mask).sel(lead_time=in_day1)
    manual = np.sqrt(err2.mean(["init_time", "lead_time"], skipna=True))

    first_bucket = out["rmse"].sel(tod="day", day=1)
    np.testing.assert_allclose(first_bucket.values, manual.values, rtol=1e-5)
    assert out["day_label"].values[0] == "≤ Day 1"


def test_align_baseline_persistence_hand_check():
    pred, obs = _synthetic_cube()
    valid_times = pd.date_range("2024-01-01", periods=60, freq="1h")
    series = xr.DataArray(
        np.arange(60, dtype=float)[:, None, None] + np.zeros((60, 3, 2)),
        dims=("valid_time", "lat", "lon"),
        coords={"valid_time": valid_times, "lat": pred["lat"], "lon": pred["lon"]},
    )
    cube = align_baseline(series, pred)
    assert cube.dims == pred.dims
    # init 2024-01-01 06:00 + lead 3h -> valid index 9 -> value 9, wherever pred is finite
    sample = cube.sel(init_time="2024-01-01T06:00", lead_time=np.timedelta64(3, "h"))
    finite = np.isfinite(sample.values)
    assert finite.any()
    np.testing.assert_allclose(sample.values[finite], 9.0)


def test_products_on_lazy_inputs():
    # The full pipeline runs dask-backed; eager-only testing missed a boolean-dask
    # indexer crash in the valid-day pooling. Exercise every pooling mode lazily.
    pred, obs = _synthetic_cube()
    pred_lazy = pred.chunk({"init_time": 4})
    obs_lazy = obs.chunk({"init_time": 4})
    baseline = _baseline_cube(obs).chunk({"init_time": 4})

    out = compute_products(
        pred_lazy,
        obs_lazy,
        [
            "scores_by_lead",
            "scores_by_forecast_day",
            "rmse_by_valid_day_lead1_24",
            "rmse_map_by_bucket_tod",
        ],
        baselines={"pers": baseline},
    )
    for name, product in out.items():
        computed = product.compute()
        assert all(bool(computed[v].notnull().any()) for v in ("rmse",)), name

    eager = compute_products(pred, obs, ["rmse_by_valid_day_lead1_24"])["rmse_by_valid_day_lead1_24"]
    lazy = out["rmse_by_valid_day_lead1_24"].compute()
    np.testing.assert_allclose(lazy["rmse"].values, eager["rmse"].values, rtol=1e-6)
