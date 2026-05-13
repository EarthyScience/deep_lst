"""Metric computation for forecast evaluation.

All scoring operates on xr.DataArrays with dim ``valid_time`` (and optionally
``lat``, ``lon``). Use ``to_valid_time()`` to convert a ForecastResult dataset
from ``(init_time, lead_time, lat, lon)`` to ``(valid_time, lat, lon)`` before
calling ``compute_scores()``.

Aggregation scenarios
---------------------
Controlled by the ``dims`` argument to ``compute_scores()``:

  "total"         → reduce (valid_time, lat, lon) → scalar per metric
  "region_time"   → reduce (lat, lon) only        → shape (valid_time,)
  "region_space"  → reduce (valid_time,) only     → shape (lat, lon)
  "none"          → no reduction                  → shape (valid_time, lat, lon)

ERA5 baseline (case C) uses ``valid_time`` directly; model forecasts (A, B)
must be converted first with ``to_valid_time()``.
"""

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Available metrics
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, callable] = {}


def _register(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


@_register("bias")
def _bias(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    return (pred - obs).mean(dim=dim, skipna=True)


@_register("rmse")
def _rmse(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    return np.sqrt(((obs - pred) ** 2).mean(dim=dim, skipna=True))


@_register("mae")
def _mae(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    return np.abs(obs - pred).mean(dim=dim, skipna=True)


@_register("anomaly_correlation")
def _anomaly_correlation(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    """Pearson r of anomalies (pred-pred_mean, obs-obs_mean) to get ACC."""
    p = pred - pred.mean(dim=dim, skipna=True)
    o = obs - obs.mean(dim=dim, skipna=True)
    num = (p * o).sum(dim=dim, skipna=True)
    denom = np.sqrt((p**2).sum(dim=dim, skipna=True) * (o**2).sum(dim=dim, skipna=True))
    return num / denom


@_register("r2")
def _r2(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    ss_res = ((obs - pred) ** 2).sum(dim=dim, skipna=True)
    obs_mean = obs.mean(dim=dim, skipna=True)
    ss_tot = ((obs - obs_mean) ** 2).sum(dim=dim, skipna=True)
    return 1.0 - ss_res / ss_tot


@_register("nash_sufcliffe_efficiency")
def _nash_sufcliffe_efficiency(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    """Nash-Sutcliffe Efficiency (NSE) = 1 - (SS_res / SS_tot), where:
    SS_res = sum((obs - pred)²) and SS_tot = sum((obs - obs_mean)²).
    The sum is the sum over the time dimension.
    NSE ranges from -∞ to 1, where 1 indicates a perfect match between pred and obs.
    """
    ss_res = ((obs - pred) ** 2).sum(dim=dim, skipna=True)
    obs_mean = obs.mean(dim=dim, skipna=True)  # total mean of obs across the time dimension
    ss_tot = ((obs - obs_mean) ** 2).sum(dim=dim, skipna=True)
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def to_valid_time(
    ds: xr.Dataset,
    var: str,
    lead_time: str | np.timedelta64 | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Select a lead time and convert init_time → valid_time.

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult dataset with dims (init_time, lead_time, lat, lon).
        Must contain ``{var}_pred`` and ``{var}_obs``.
    var : str
        Variable name, e.g. "LST".
    lead_time : str or timedelta64, optional
        Lead time to select, e.g. "1h" or np.timedelta64(1, "h").
        If None and the dataset has only one lead_time, it is used automatically
        (covers the diagnostic model which has lead_time=1h only).

    Returns
    -------
    pred, obs : xr.DataArray
        Both with dim ``valid_time`` (and lat, lon if present).
    """
    pred = ds[f"{var}_pred"]
    obs = ds[f"{var}_obs"]

    # Select lead time
    if lead_time is None:
        if pred.sizes.get("lead_time", 1) == 1:
            pred = pred.isel(lead_time=0)
            obs = obs.isel(lead_time=0)
            lt_delta = ds["lead_time"].values[0]
        else:
            raise ValueError("lead_time must be specified when the dataset has multiple lead times.")
    else:
        pred = pred.sel(lead_time=lead_time)
        obs = obs.sel(lead_time=lead_time)
        lt_delta = np.timedelta64(pd.Timedelta(lead_time).value, "ns")

    # Shift init_time → valid_time
    valid_times = ds["init_time"].values + lt_delta
    pred = pred.assign_coords(init_time=valid_times).rename({"init_time": "valid_time"})
    obs = obs.assign_coords(init_time=valid_times).rename({"init_time": "valid_time"})

    return pred, obs


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

_AGGREGATION_DIMS = {
    "total": ["valid_time", "lat", "lon", "pixel"],
    "region_time": ["lat", "lon", "pixel"],
    "region_space": ["valid_time"],
    "none": [],
}


def compute_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str] | None = None,
    aggregation: str = "total",
) -> xr.Dataset:
    """Compute metrics between pred and obs, reducing over the specified dims.

    Parameters
    ----------
    pred, obs : xr.DataArray
        Must share dims. NaN values are excluded from all computations.
        Typically on dims (valid_time, lat, lon) after ``to_valid_time()``.
        ERA5 baseline can be passed directly (already on valid_time).
    metrics : list of str, optional
        Subset of METRIC_REGISTRY keys. Default: ["rmse"].
    aggregation : str
        One of "total", "region_time", "region_space", "none".

    Returns
    -------
    xr.Dataset
        One variable per metric, reduced over the appropriate dims.
        Also contains ``n_valid`` (count of finite pairs used).
    """
    if metrics is None:
        metrics = ["rmse"]

    unknown = set(metrics) - METRIC_REGISTRY.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {list(METRIC_REGISTRY)}")

    if aggregation not in _AGGREGATION_DIMS:
        raise ValueError(f"Unknown aggregation: {aggregation!r}. Choose from {list(_AGGREGATION_DIMS)}")

    # Only reduce dims that actually exist in the arrays
    available_dims = set(pred.dims)
    dim = [d for d in _AGGREGATION_DIMS[aggregation] if d in available_dims] or None

    data_vars = {}
    for name in metrics:
        fn = METRIC_REGISTRY[name]
        data_vars[name] = fn(pred, obs, dim=dim)

    # n_valid: count of finite pairs, reduced over same dims
    valid_mask = np.isfinite(pred) & np.isfinite(obs)
    if dim:
        data_vars["n_valid"] = valid_mask.sum(dim=dim)
    else:
        data_vars["n_valid"] = valid_mask

    return xr.Dataset(data_vars)


# ---------------------------------------------------------------------------
# Validity report
# ---------------------------------------------------------------------------


def compute_validity_report(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> dict:
    """Summarise why pixel-timestep pairs are invalid.

    Distinguishes between:
    - ``n_valid``        : finite pred AND finite obs  → used in scoring
    - ``n_invalid_init`` : NaN pred (forecast couldn't start, e.g. cloudy init)
    - ``n_invalid_truth``: finite pred but NaN obs (truth cloud-covered)
    - ``n_total``        : total elements

    Parameters
    ----------
    pred, obs : xr.DataArray
        On any shared dims; typically (valid_time, lat, lon).

    Returns
    -------
    dict with integer counts.
    """
    pred_nan = ~np.isfinite(pred)
    obs_nan = ~np.isfinite(obs)

    n_total = int(pred.size)
    n_invalid_init = int((pred_nan).sum().values)
    n_invalid_truth = int((~pred_nan & obs_nan).sum().values)
    n_valid = int((~pred_nan & ~obs_nan).sum().values)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid_init": n_invalid_init,
        "n_invalid_truth": n_invalid_truth,
    }


# ---------------------------------------------------------------------------
# Diurnal cycle (kept from original, refactored to use valid_time dim)
# ---------------------------------------------------------------------------


def compute_diurnal_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str] | None = None,
) -> xr.Dataset:
    """Compute per-hour-of-day scores, aggregating over pixels.

    Parameters
    ----------
    pred, obs : xr.DataArray
        On dims (valid_time, lat, lon). Produced by ``to_valid_time()``.
    metrics : list of str, optional
        Default: ["bias", "rmse"].

    Returns
    -------
    xr.Dataset with dim ``hour`` (0–23), one variable per metric + ``n_valid``.
    """
    if metrics is None:
        metrics = ["bias", "rmse"]

    hours = pred["valid_time"].dt.hour.values  # (n_valid_time,)
    results = {m: np.full(24, np.nan) for m in metrics}
    std_by_hour = np.full(24, np.nan)
    n_valid = np.zeros(24, dtype=np.int64)

    pred_vals = pred.transpose("valid_time", ...).values
    obs_vals = obs.transpose("valid_time", ...).values

    for h in range(24):
        h_mask = hours == h  # (n_valid_time,)
        p_h = pred_vals[h_mask]  # (n_h, lat, lon) or (n_h, pixel)
        o_h = obs_vals[h_mask]
        valid = np.isfinite(p_h) & np.isfinite(o_h)
        n = int(valid.sum())
        n_valid[h] = n
        if n == 0:
            continue
        p_v, o_v = p_h[valid], o_h[valid]
        for m in metrics:
            if m == "bias":
                results[m][h] = float(np.mean(o_v - p_v))
            elif m == "rmse":
                results[m][h] = float(np.sqrt(np.mean((o_v - p_v) ** 2)))
            elif m == "mae":
                results[m][h] = float(np.mean(np.abs(o_v - p_v)))
            elif m == "r2":
                ss_res = np.sum((o_v - p_v) ** 2)
                ss_tot = np.sum((o_v - np.mean(o_v)) ** 2)
                results[m][h] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        std_by_hour[h] = float(np.std(o_v - p_v))

    data_vars = {m: ("hour", results[m]) for m in metrics}
    data_vars["std"] = ("hour", std_by_hour)
    data_vars["n_valid"] = ("hour", n_valid)

    return xr.Dataset(data_vars, coords={"hour": np.arange(24)})


def compute_diurnal_mean(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> xr.Dataset:
    """Compute per-hour-of-day mean and std of raw pred and obs values.

    Used for diurnal mean and diurnal delta (ΔT) plots. Unlike
    compute_diurnal_scores, this operates on raw values rather than errors.

    Parameters
    ----------
    pred, obs : xr.DataArray
        On dims (valid_time, lat, lon). Produced by to_valid_time().

    Returns
    -------
    xr.Dataset with dim hour (0–23) and variables:
        pred_mean, obs_mean       : mean LST per hour
        pred_std,  obs_std        : spatial+temporal std per hour
        pred_delta, obs_delta     : hour-to-hour change (circular, K/h)
        n_valid                   : number of finite pixel-timestep pairs
    """
    hours = pred["valid_time"].dt.hour.values
    pred_mean = np.full(24, np.nan)
    obs_mean = np.full(24, np.nan)
    pred_std = np.full(24, np.nan)
    obs_std = np.full(24, np.nan)
    n_valid = np.zeros(24, dtype=np.int64)

    pred_vals = pred.transpose("valid_time", ...).values
    obs_vals = obs.transpose("valid_time", ...).values
    # Flatten spatial dims so each column is one pixel: (valid_time, n_pixels)
    pred_flat = pred_vals.reshape(len(hours), -1)
    obs_flat = obs_vals.reshape(len(hours), -1)

    # Per-pixel hourly means — shape (24, n_pixels)
    pred_hourly_px = np.full((24, pred_flat.shape[1]), np.nan)
    obs_hourly_px = np.full((24, pred_flat.shape[1]), np.nan)

    for h in range(24):
        h_mask = hours == h
        p_h = pred_flat[h_mask]
        o_h = obs_flat[h_mask]
        valid = np.isfinite(p_h) & np.isfinite(o_h)
        n = int(valid.sum())
        n_valid[h] = n
        if n == 0:
            continue
        p_h_masked = np.where(valid, p_h, np.nan)
        o_h_masked = np.where(valid, o_h, np.nan)
        pred_mean[h] = float(np.nanmean(p_h_masked))
        obs_mean[h] = float(np.nanmean(o_h_masked))
        pred_std[h] = float(np.nanstd(p_h_masked))
        obs_std[h] = float(np.nanstd(o_h_masked))
        pred_hourly_px[h] = np.nanmean(p_h_masked, axis=0)
        obs_hourly_px[h] = np.nanmean(o_h_masked, axis=0)

    # Hour-to-hour change: delta[h] = mean[h] - mean[h-1] (circular)
    pred_delta = np.roll(pred_mean, -1) - pred_mean  # delta[h] = mean[h+1] - mean[h]
    obs_delta = np.roll(obs_mean, -1) - obs_mean
    # Where either neighbour is NaN the diff is meaningless
    for arr in (pred_delta, obs_delta):
        for h in range(24):
            if np.isnan(pred_mean[h]) or np.isnan(pred_mean[(h + 1) % 24]):
                arr[h] = np.nan

    # Per-pixel delta error: (pred_delta_px - obs_delta_px), shape (24, n_pixels)
    pred_delta_px = np.roll(pred_hourly_px, -1, axis=0) - pred_hourly_px
    obs_delta_px = np.roll(obs_hourly_px, -1, axis=0) - obs_hourly_px
    delta_err_px = pred_delta_px - obs_delta_px
    # Mask hours where either neighbour is NaN for any pixel
    for h in range(24):
        if np.isnan(pred_mean[h]) or np.isnan(pred_mean[(h + 1) % 24]):
            delta_err_px[h] = np.nan

    delta_err_q05 = np.nanpercentile(delta_err_px, 5, axis=1)
    delta_err_q50 = np.nanpercentile(delta_err_px, 50, axis=1)
    delta_err_q95 = np.nanpercentile(delta_err_px, 95, axis=1)

    return xr.Dataset(
        {
            "pred_mean": ("hour", pred_mean),
            "obs_mean": ("hour", obs_mean),
            "pred_std": ("hour", pred_std),
            "obs_std": ("hour", obs_std),
            "pred_delta": ("hour", pred_delta),
            "obs_delta": ("hour", obs_delta),
            "delta_err_q05": ("hour", delta_err_q05),
            "delta_err_q50": ("hour", delta_err_q50),
            "delta_err_q95": ("hour", delta_err_q95),
            "n_valid": ("hour", n_valid),
        },
        coords={"hour": np.arange(24)},
    )


def compute_skill_scores_by_lead_time(
    ds: xr.Dataset,
    ref_da: xr.DataArray,
    ref_type: str,
    var: str = "LST",
    metrics: list[str] | None = None,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
) -> xr.Dataset:
    """Compute skill score (model / reference) per lead time with empirical percentile CI.

    Per-time-step skill = spatial_RMSE_model(t) / spatial_RMSE_ref(t) is computed
    directly so that percentiles are of the actual skill distribution, not an
    approximation from separate numerator/denominator percentiles.

    Central estimate is the median (q50) for consistency with the asymmetric CI.

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult with dims (init_time, lead_time, ...).
    ref_da : xr.DataArray
        Reference DataArray with dim ``valid_time``.
    ref_type : str
        ``"persistence"`` looks up ref_da at init_time (valid_time − lt).
        ``"climatology"`` looks up ref_da at valid_time directly.
    q_lo, q_hi : float
        Lower and upper quantiles for the CI band (default: 0.05 / 0.95).
    """
    if metrics is None:
        metrics = ["rmse"]

    lead_times = ds["lead_time"].values
    q_lo_label = f"q{int(q_lo * 100):02d}"
    q_hi_label = f"q{int(q_hi * 100):02d}"
    results_median = {m: [] for m in metrics}
    results_q_lo = {m: [] for m in metrics}
    results_q_hi = {m: [] for m in metrics}
    n_valid_list = []

    for lt in lead_times:
        lt_ns = np.timedelta64(int(pd.Timedelta(lt).total_seconds() * 1e9), "ns")
        lt_h = int(pd.Timedelta(lt).total_seconds() // 3600)
        try:
            pred, obs = to_valid_time(ds, var, lead_time=f"{lt_h}h")
        except Exception:
            for m in metrics:
                results_median[m].append(np.nan)
                results_q_lo[m].append(np.nan)
                results_q_hi[m].append(np.nan)
            n_valid_list.append(0)
            continue

        if ref_type == "persistence":
            init_times = obs["valid_time"].values - lt_ns
            ref = ref_da.reindex(valid_time=init_times)
            ref = ref.assign_coords(valid_time=obs["valid_time"].values)
        else:
            ref = ref_da.reindex(valid_time=obs["valid_time"].values)
        ref = ref.where(np.isfinite(pred))

        model_per_time = compute_scores(pred, obs, metrics=metrics, aggregation="region_time")
        ref_per_time = compute_scores(ref, obs, metrics=metrics, aggregation="region_time")

        for m in metrics:
            skill_per_time = model_per_time[m] / ref_per_time[m]
            if skill_per_time.isnull().all():
                results_median[m].append(np.nan)
                results_q_lo[m].append(np.nan)
                results_q_hi[m].append(np.nan)
            else:
                results_median[m].append(float(skill_per_time.quantile(0.5).values))
                results_q_lo[m].append(float(skill_per_time.quantile(q_lo).values))
                results_q_hi[m].append(float(skill_per_time.quantile(q_hi).values))
        n_valid_list.append(int(model_per_time["n_valid"].sum().values))

    return xr.Dataset(
        {m: ("lead_time", results_median[m]) for m in metrics}
        | {f"{m}_{q_lo_label}": ("lead_time", results_q_lo[m]) for m in metrics}
        | {f"{m}_{q_hi_label}": ("lead_time", results_q_hi[m]) for m in metrics}
        | {"n_valid": ("lead_time", n_valid_list)},
        coords={"lead_time": lead_times},
    )


def compute_persistence_scores_by_lead_time(
    ds: xr.Dataset,
    pers_da: xr.DataArray,
    var: str = "LST",
    metrics: list[str] | None = None,
) -> xr.Dataset:
    """Compute persistence baseline metrics at every lead_time.

    For each lead time lt, persistence predicts obs(init_time) = obs(valid_time - lt).
    Metrics are aggregated over all (init_time, lat, lon / pixel).

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult with dims (init_time, lead_time, ...).
    pers_da : xr.DataArray
        Raw persistence file DataArray with dim ``valid_time`` → obs(t).
    """
    if metrics is None:
        metrics = ["rmse"]

    lead_times = ds["lead_time"].values
    results = {m: [] for m in metrics}
    results_std = {m: [] for m in metrics}
    n_valid_list = []

    for lt in lead_times:
        lt_ns = np.timedelta64(int(pd.Timedelta(lt).total_seconds() * 1e9), "ns")
        lt_h = int(pd.Timedelta(lt).total_seconds() // 3600)
        try:
            pred, obs = to_valid_time(ds, var, lead_time=f"{lt_h}h")
        except Exception:
            for m in metrics:
                results[m].append(np.nan)
                results_std[m].append(np.nan)
            n_valid_list.append(0)
            continue

        init_times = obs["valid_time"].values - lt_ns
        pers = pers_da.reindex(valid_time=init_times)
        pers = pers.assign_coords(valid_time=obs["valid_time"].values)
        pers = pers.where(np.isfinite(pred))

        scores_per_time = compute_scores(pers, obs, metrics=metrics, aggregation="region_time")
        for m in metrics:
            results[m].append(float(scores_per_time[m].mean().values))
            results_std[m].append(float(scores_per_time[m].std().values))
        n_valid_list.append(int(scores_per_time["n_valid"].sum().values))

    return xr.Dataset(
        {m: ("lead_time", results[m]) for m in metrics}
        | {f"{m}_std": ("lead_time", results_std[m]) for m in metrics}
        | {"n_valid": ("lead_time", n_valid_list)},
        coords={"lead_time": lead_times},
    )


def compute_climatology_scores_by_lead_time(
    ds: xr.Dataset,
    clim_da: xr.DataArray,
    var: str = "LST",
    metrics: list[str] | None = None,
) -> xr.Dataset:
    """Compute climatology baseline metrics at every lead_time.

    For each lead time lt, climatology predicts clim(valid_time).
    Unlike persistence, this does not depend on init_time.

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult with dims (init_time, lead_time, ...).
    clim_da : xr.DataArray
        Climatology DataArray with dim ``valid_time``.
    """
    if metrics is None:
        metrics = ["rmse"]

    lead_times = ds["lead_time"].values
    results = {m: [] for m in metrics}
    results_std = {m: [] for m in metrics}
    n_valid_list = []

    for lt in lead_times:
        lt_h = int(pd.Timedelta(lt).total_seconds() // 3600)
        try:
            pred, obs = to_valid_time(ds, var, lead_time=f"{lt_h}h")
        except Exception:
            for m in metrics:
                results[m].append(np.nan)
                results_std[m].append(np.nan)
            n_valid_list.append(0)
            continue

        clim = clim_da.reindex(valid_time=obs["valid_time"].values)
        if lt_h == int(pd.Timedelta(lead_times[0]).total_seconds() // 3600):
            n_before = int(np.isfinite(clim).sum().values)
            n_pred_valid = int(np.isfinite(pred).sum().values)
        clim = clim.where(np.isfinite(pred))
        if lt_h == int(pd.Timedelta(lead_times[0]).total_seconds() // 3600):
            n_after = int(np.isfinite(clim).sum().values)
            print(
                f"  [clim mask check lt={lt_h}h] clim valid: {n_before}, pred valid: {n_pred_valid}, after masking: {n_after} (removed {n_before - n_after})"
            )

        scores_per_time = compute_scores(clim, obs, metrics=metrics, aggregation="region_time")
        for m in metrics:
            results[m].append(float(scores_per_time[m].mean().values))
            results_std[m].append(float(scores_per_time[m].std().values))
        n_valid_list.append(int(scores_per_time["n_valid"].sum().values))

    return xr.Dataset(
        {m: ("lead_time", results[m]) for m in metrics}
        | {f"{m}_std": ("lead_time", results_std[m]) for m in metrics}
        | {"n_valid": ("lead_time", n_valid_list)},
        coords={"lead_time": lead_times},
    )


def compute_scores_by_lead_time(
    ds: xr.Dataset,
    var: str = "LST",
    metrics: list[str] | None = None,
) -> xr.Dataset:
    """Compute metrics at every lead_time, aggregated over all (init_time, lat, lon).

    Returns a Dataset with dim ``lead_time`` suitable for error-growth plots
    and for saving as ``metrics_by_lead_time.nc``.

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult with dims (init_time, lead_time, lat, lon).
    var : str
        Variable name, e.g. "LST".
    metrics : list of str, optional
        Default: ["rmse", "bias"].

    Returns
    -------
    xr.Dataset with dim ``lead_time`` and one variable per metric + ``n_valid``.
    """
    if metrics is None:
        metrics = ["rmse", "bias"]

    lead_times = ds["lead_time"].values
    results = {m: [] for m in metrics}
    results_std = {m: [] for m in metrics}
    n_valid_list = []

    for lt in lead_times:
        lt_h = int(pd.Timedelta(lt).total_seconds() // 3600)
        try:
            pred, obs = to_valid_time(ds, var, lead_time=f"{lt_h}h")
        except Exception:
            for m in metrics:
                results[m].append(np.nan)
                results_std[m].append(np.nan)
            n_valid_list.append(0)
            continue

        scores_per_time = compute_scores(pred, obs, metrics=metrics, aggregation="region_time")
        for m in metrics:
            results[m].append(float(scores_per_time[m].mean().values))
            results_std[m].append(float(scores_per_time[m].std().values))
        n_valid_list.append(int(scores_per_time["n_valid"].sum().values))

    return xr.Dataset(
        {m: ("lead_time", results[m]) for m in metrics}
        | {f"{m}_std": ("lead_time", results_std[m]) for m in metrics}
        | {"n_valid": ("lead_time", n_valid_list)},
        coords={"lead_time": lead_times},
    )
