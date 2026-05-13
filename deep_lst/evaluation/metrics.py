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

_GROUPBY = {
    "diurnal": (lambda da: da["valid_time"].dt.hour.values, "hour", np.arange(24)),
    "monthly": (lambda da: da["valid_time"].dt.month.values, "month", np.arange(1, 13)),
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


def compute_grouped_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str] | None = None,
    aggregation: str = "total",
    groupby: str = "diurnal",
) -> xr.Dataset:
    get_group, dim_name, dim_values = _GROUPBY[groupby]
    group_idx = get_group(pred)
    slices = []
    for g in dim_values:
        idx = np.where(group_idx == g)[0]
        p_g = pred.isel(valid_time=idx)
        o_g = obs.isel(valid_time=idx)
        ds_g = compute_scores(p_g, o_g, metrics, aggregation=aggregation)
        dim = [d for d in _AGGREGATION_DIMS[aggregation] if d in set(p_g.dims)] or None
        ds_g = ds_g.assign({"pred_mean": p_g.mean(dim=dim, skipna=True), "obs_mean": o_g.mean(dim=dim, skipna=True)})
        slices.append(ds_g)

    ds = xr.concat(slices, dim=pd.Index(dim_values, name=dim_name))

    return ds


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
