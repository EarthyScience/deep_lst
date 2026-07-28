import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.conditioning import time_of_day_mask
from deep_lst.evaluation.metrics import skill_score
from deep_lst.evaluation.products import align_baseline, compute_products

RAW = ["rmse", "mae", "bias", "nse"]


def _ref_rmse(pred, obs, dim=None):
    """Independent reference: joint-masked pooled RMSE (mirrors the deleted registry formula)."""
    return np.sqrt(((obs - pred) ** 2).mean(dim=dim, skipna=True))


def _ref_bias(pred, obs, dim=None):
    return (pred - obs).mean(dim=dim, skipna=True)


def _ref_acc(anom_pred, anom_obs, dim=None):
    valid = anom_pred.notnull() & anom_obs.notnull()
    ap, ao = anom_pred.where(valid), anom_obs.where(valid)
    num = (ap * ao).sum(dim=dim, skipna=True)
    return num / np.sqrt((ap**2).sum(dim=dim, skipna=True) * (ao**2).sum(dim=dim, skipna=True))


def _ref_activity(anom_pred, anom_obs, dim=None):
    return anom_pred.where(anom_obs.notnull()).std(dim=dim, skipna=True)


def _ref_mae(pred, obs, dim=None):
    return np.abs(obs - pred).mean(dim=dim, skipna=True)


def _ref_nse(pred, obs, dim=None):
    valid = pred.notnull() & obs.notnull()
    p, o = pred.where(valid), obs.where(valid)
    ss_res = ((o - p) ** 2).sum(dim=dim, skipna=True)
    ss_tot = ((o - o.mean(dim=dim, skipna=True)) ** 2).sum(dim=dim, skipna=True)
    return 1.0 - ss_res / ss_tot


_REFS = {"rmse": _ref_rmse, "bias": _ref_bias, "mae": _ref_mae, "nse": _ref_nse}


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


def test_scores_by_lead_matches_inline_references():
    pred, obs = _synthetic_cube()
    out = compute_products(pred, obs, ["scores_by_lead"])["scores_by_lead"]
    pooled_dims = ["init_time", "lat", "lon"]
    for m in RAW:
        reference = _REFS[m](pred, obs, dim=pooled_dims)
        np.testing.assert_allclose(out[m].values, reference.values, rtol=1e-4, equal_nan=True)


def test_scores_by_forecast_day_matches_and_skill():
    pred, obs = _synthetic_cube()
    baseline = _baseline_cube(obs)
    out = compute_products(pred, obs, ["scores_by_forecast_day"], baselines={"pers": baseline})
    product = out["scores_by_forecast_day"]

    lead_hours = (pred["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    for i, day in enumerate(product["day"].values):
        in_bucket = (lead_hours >= (day - 1) * 24 + 1) & (lead_hours <= day * 24)
        pred_b = pred.isel(lead_time=np.where(in_bucket)[0])
        obs_b = obs.isel(lead_time=np.where(in_bucket)[0])
        assert np.isclose(float(product["rmse"].isel(day=i)), float(_ref_rmse(pred_b, obs_b)), rtol=1e-4)
        assert np.isclose(
            float(product["rmse_pers"].isel(day=i)),
            float(_ref_rmse(baseline.isel(lead_time=np.where(in_bucket)[0]), obs_b)),
            rtol=1e-4,
        )
    expected_skill = skill_score(product["rmse"], product["rmse_pers"])
    np.testing.assert_allclose(product["rmse_skill_pers"].values, expected_skill.values, rtol=1e-6)
    assert list(product["day_label"].values) == [f"Day {d}" for d in product["day"].values]


def test_valid_day_product_matches_registry_per_day():
    from deep_lst.evaluation.conditioning import valid_day_labels

    pred, obs = _synthetic_cube()
    pred_win = pred.isel(lead_time=slice(0, 24))
    obs_win = obs.isel(lead_time=slice(0, 24))
    out = compute_products(pred, obs, ["rmse_by_valid_day_lead1_24"])["rmse_by_valid_day_lead1_24"]

    labels = valid_day_labels(pred_win)
    for i, day in enumerate(out["day"].values):
        cell_mask = labels == day
        ref = float(_ref_rmse(pred_win.where(cell_mask), obs_win.where(cell_mask), dim=None))
        assert np.isclose(float(out["rmse"].isel(day=i)), ref, rtol=1e-5), str(day)


def test_anomaly_products_match_registry():
    pred, obs = _synthetic_cube()
    clim = obs * 0 + 300.0  # any shared reference works for the identity
    out = compute_products(pred, obs, ["anomaly_by_init_lead"], clim=clim)["anomaly_by_init_lead"]

    ap, ao = pred - clim, obs - clim
    old = xr.Dataset(
        {m: {"acc": _ref_acc, "activity": _ref_activity}[m](ap, ao, dim=["lat", "lon"]) for m in ("acc", "activity")}
    )
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


# --- diagnostic products (the single-lead valid_time view, Diagnostics fold) ---


def _diag_view(seed=7):
    """Single-lead (valid_time, lat, lon) view with a class coord, plus the raw pair."""
    from deep_lst.evaluation.metrics import to_valid_time

    pred, obs = _synthetic_cube(seed)
    lead = np.timedelta64(3, "h")
    pred_lead = to_valid_time(pred.sel(lead_time=lead), "3h")
    obs_lead = to_valid_time(obs.sel(lead_time=lead), "3h")
    kg = xr.DataArray([[1, 2], [1, -1], [2, 2]], dims=("lat", "lon"))
    pred_lead = pred_lead.assign_coords(kg_class=(("lat", "lon"), kg.values))
    obs_lead = obs_lead.assign_coords(kg_class=(("lat", "lon"), kg.values))
    return pred_lead, obs_lead


def test_diag_map_and_by_time_match_registry():
    from deep_lst.evaluation.products import compute_diagnostic_products

    pred, obs = _diag_view()
    out = compute_diagnostic_products(pred, obs, ["diag_map", "diag_scores_by_time"])

    for m in RAW:
        ref_map = _REFS[m](pred, obs, dim=["valid_time"])
        np.testing.assert_allclose(out["diag_map"][m].values, ref_map.values, rtol=1e-4, equal_nan=True)
        ref_time = _REFS[m](pred, obs, dim=["lat", "lon"])
        np.testing.assert_allclose(out["diag_scores_by_time"][m].values, ref_time.values, rtol=1e-4, equal_nan=True)


def test_diag_by_hour_matches_registry_per_hour():
    from deep_lst.evaluation.products import compute_diagnostic_products

    pred, obs = _diag_view()
    out = compute_diagnostic_products(pred, obs, ["diag_by_hour"])["diag_by_hour"]
    assert out.sizes["hour"] == 24

    hours = pred["valid_time"].dt.hour.values
    valid = pred.notnull() & obs.notnull()
    for hour in np.unique(hours):
        idx = np.where(hours == hour)[0]
        pred_h = pred.isel(valid_time=idx)
        obs_h = obs.isel(valid_time=idx)
        ref = _ref_rmse(pred_h, obs_h, dim=["valid_time"])
        np.testing.assert_allclose(out["rmse"].sel(hour=hour).values, ref.values, rtol=1e-4, equal_nan=True)
        ref_mean = pred_h.where(valid.isel(valid_time=idx)).mean("valid_time", skipna=True)
        np.testing.assert_allclose(out["pred_mean"].sel(hour=hour).values, ref_mean.values, rtol=1e-4, equal_nan=True)
    absent = sorted(set(range(24)) - set(hours))
    if absent:
        assert bool(out["rmse"].sel(hour=absent).isnull().all())


def test_diag_by_class_matches_registry_per_class():
    from deep_lst.evaluation.products import compute_diagnostic_products

    pred, obs = _diag_view()
    out = compute_diagnostic_products(pred, obs, ["diag_by_class"], class_coords=("kg_class",))
    product = out["diag_by_class__kg_class"]

    assert -1 not in product["kg_class"].values  # nodata class excluded
    for class_id in (1, 2):
        mask = pred["kg_class"] == class_id
        ref = _ref_rmse(pred.where(mask), obs.where(mask), dim=["lat", "lon"])
        np.testing.assert_allclose(product["rmse"].sel(kg_class=class_id).values, ref.values, rtol=1e-4, equal_nan=True)


def test_diag_by_day_matches_registry_per_day():
    from deep_lst.evaluation.products import compute_diagnostic_products

    pred, obs = _diag_view()
    out = compute_diagnostic_products(pred, obs, ["diag_by_day"])["diag_by_day"]

    days = pred["valid_time"].dt.floor("D").values
    for metric in ("rmse", "bias"):
        for i, day in enumerate(np.unique(days)):
            idx = np.where(days == day)[0]
            ref = float(
                {"rmse": _ref_rmse, "bias": _ref_bias}[metric](
                    pred.isel(valid_time=idx), obs.isel(valid_time=idx), dim=None
                )
            )
            assert np.isclose(float(out[metric].isel(day=i)), ref, rtol=1e-4)
        total = float({"rmse": _ref_rmse, "bias": _ref_bias}[metric](pred, obs, dim=None))
        assert np.isclose(float(out[f"{metric}_total"]), total, rtol=1e-4)


def test_diag_baseline_columns_and_tod():
    from deep_lst.evaluation.conditioning import time_of_day_mask
    from deep_lst.evaluation.metrics import skill_score
    from deep_lst.evaluation.products import compute_diagnostic_products

    pred, obs = _diag_view()
    baseline = _baseline_cube(obs)

    out = compute_diagnostic_products(pred, obs, ["diag_map"], baselines={"pers": baseline})["diag_map"]
    old_bl = _ref_rmse(baseline, obs, dim=["valid_time"])
    np.testing.assert_allclose(out["rmse_pers"].values, old_bl.values, rtol=1e-4, equal_nan=True)
    expected_skill = skill_score(out["rmse"], out["rmse_pers"])
    np.testing.assert_allclose(out["rmse_skill_pers"].values, expected_skill.values, rtol=1e-6, equal_nan=True)

    # tod conditioning == masking the inputs up front
    day = compute_diagnostic_products(pred, obs, ["diag_scores_by_time"], tod="day")["diag_scores_by_time"]
    window = time_of_day_mask(pred, "day")
    old_day = _ref_rmse(pred.where(window), obs.where(window), dim=["lat", "lon"])
    np.testing.assert_allclose(day["rmse"].values, old_day.values, rtol=1e-4, equal_nan=True)
