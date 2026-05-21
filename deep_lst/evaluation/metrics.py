"""Metric computation for forecast evaluation."""

import numpy as np
import pandas as pd
import xarray as xr

METRIC_REGISTRY: dict[str, callable] = {}

_AGGREGATION_DIMS = {
    "total": ["valid_time", "init_time", "lat", "lon", "pixel"],
    "region_time": ["lat", "lon", "pixel"],
    "region_space": ["valid_time", "init_time"],
    "none": [],
}

_REDUCTION_DIMS = {
    "time": ["valid_time", "init_time"],
    "space": ["lat", "lon", "pixel"],
}

_GROUPBY = {
    "diurnal": (lambda da: da.dt.hour.values, "hour", np.arange(24)),
    "monthly": (lambda da: da.dt.month.values, "month", np.arange(1, 13)),
}


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


@_register("activity")
def _activity(anom_pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    """Std of pred over dim. Expects pred_anom = pred - clim as input."""
    return anom_pred.std(dim=dim, skipna=True)


@_register("acc")
def _acc(anom_pred: xr.DataArray, anom_obs: xr.DataArray, dim) -> xr.DataArray:
    """Uncentred correlation for pre-computed anomalies (proper ACC).
    Expects pred_anom and obs_anom as inputs — does not subtract mean internally."""
    num = (anom_pred * anom_obs).sum(dim=dim, skipna=True)
    denom = np.sqrt((anom_pred**2).sum(dim=dim, skipna=True) * (anom_obs**2).sum(dim=dim, skipna=True))
    return num / denom


@_register("c_acc")
def c_acc(anom_pred: xr.DataArray, anom_obs: xr.DataArray, dim) -> xr.DataArray:
    """Pearson r of anomalies of pred and obs (centres internally). Use for raw time series."""
    p = anom_pred - anom_pred.mean(dim=dim, skipna=True)
    o = anom_obs - anom_obs.mean(dim=dim, skipna=True)
    num = (p * o).sum(dim=dim, skipna=True)
    denom = np.sqrt((p**2).sum(dim=dim, skipna=True) * (o**2).sum(dim=dim, skipna=True))
    return num / denom


@_register("nse")
def nse(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
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
# Time coordinate conversion
# ---------------------------------------------------------------------------


def to_valid_time(da: xr.Dataset, lead_time: str | np.timedelta64 | None = None) -> xr.DataArray:
    """Select a lead time and convert init_time → valid_time."""
    lt_delta = np.timedelta64(pd.Timedelta(lead_time).value, "ns")
    valid_times = da["init_time"].values + lt_delta
    return da.assign_coords(init_time=valid_times).rename({"init_time": "valid_time"})


def to_init_time(da: xr.Dataset, lead_time: str | np.timedelta64 | None = None) -> xr.DataArray:
    """Select a lead time and convert valid_time → init_time."""
    lt_delta = np.timedelta64(pd.Timedelta(lead_time).value, "ns")
    init_times = da["valid_time"].values - lt_delta
    return da.assign_coords(valid_time=init_times).rename({"valid_time": "init_time"})


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def compute_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str] | None = None,
    aggregation: str = "total",
    reduction_order: tuple = ("space", "time"),
) -> xr.Dataset:
    """Compute metrics between pred and obs, reducing over the specified dims."""
    available_dims = set(pred.dims)

    if aggregation == "total":
        first_key, second_key = reduction_order
        first_dims = [d for d in _REDUCTION_DIMS[first_key] if d in available_dims] or None
        second_dims = [d for d in _REDUCTION_DIMS[second_key] if d in available_dims]

        data_vars = {}
        for name in metrics:
            intermediate = METRIC_REGISTRY[name](pred, obs, dim=first_dims)
            remaining = [d for d in second_dims if d in intermediate.dims]
            data_vars[name] = intermediate.mean(dim=remaining or None, skipna=True)
    else:
        dim = [d for d in _AGGREGATION_DIMS[aggregation] if d in available_dims] or None
        data_vars = {}
        for name in metrics:
            data_vars[name] = METRIC_REGISTRY[name](pred, obs, dim=dim)

    return xr.Dataset(data_vars)


def compute_grouped_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str] | None = None,
    aggregation: str = "total",
    groupby: str = "diurnal",
    reduction_order: tuple = ("space", "time"),
) -> xr.Dataset:
    # detect time dimension (valid_time or init_time) for grouping
    time_dim = next(dim for dim in ("valid_time", "init_time") if dim in pred.dims)
    get_group, dim_name, dim_values = _GROUPBY[groupby]
    group_idx = get_group(pred[time_dim])

    slices = []
    for group_bin in dim_values:
        idx = np.where(group_idx == group_bin)[0]
        pred_group_bin = pred.isel({time_dim: idx})
        obs_group_bin = obs.isel({time_dim: idx})

        group_bin_scores = compute_scores(
            pred_group_bin, obs_group_bin, metrics, aggregation=aggregation, reduction_order=reduction_order
        )
        dim = [d for d in _AGGREGATION_DIMS[aggregation] if d in set(pred_group_bin.dims)] or None
        group_bin_scores = group_bin_scores.assign(
            {
                "pred_mean": pred_group_bin.mean(dim=dim, skipna=True),
                "obs_mean": obs_group_bin.mean(dim=dim, skipna=True),
            }
        )
        slices.append(group_bin_scores)

    ds = xr.concat(slices, dim=pd.Index(dim_values, name=dim_name))

    return ds
