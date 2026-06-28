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

# Inclusive lead-hour bounds of each forecast day (24 h blocks). Single source of truth shared by
# the pooled-by-day scores here and the day/night plots.
LEAD_DAY_BUCKETS = [
    (1, 24),
    (25, 48),
    (49, 72),
    (73, 96),
    (97, 120),
    (121, 144),
    (145, 168),
    (169, 192),
    (193, 216),
    (217, 240),
]


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
def _activity(anom_pred: xr.DataArray, anom_obs: xr.DataArray, dim) -> xr.DataArray:
    """Std of pred anomalies over dim, on the obs-valid samples (same masking as ACC, so activity
    and ACC are computed over the same points). Expects pred_anom, obs_anom as input."""
    return anom_pred.where(anom_obs.notnull()).std(dim=dim, skipna=True)


@_register("acc")
def _acc(anom_pred: xr.DataArray, anom_obs: xr.DataArray, dim) -> xr.DataArray:
    """Uncentred correlation for pre-computed anomalies (proper ACC).
    Expects pred_anom and obs_anom as inputs — does not subtract mean internally.

    Both anomalies are masked to their common valid samples first, so all three sums pool the
    SAME points; otherwise obs gaps (clouds) drop terms from the obs sum but not the pred sum,
    inflating the denominator and biasing ACC low."""
    valid = anom_pred.notnull() & anom_obs.notnull()
    ap = anom_pred.where(valid)
    ao = anom_obs.where(valid)
    num = (ap * ao).sum(dim=dim, skipna=True)
    denom = np.sqrt((ap**2).sum(dim=dim, skipna=True) * (ao**2).sum(dim=dim, skipna=True))
    return num / denom


@_register("c_acc")
def c_acc(anom_pred: xr.DataArray, anom_obs: xr.DataArray, dim) -> xr.DataArray:
    """Pearson r of anomalies of pred and obs (centres internally). Use for raw time series.
    Both anomalies are masked to their common valid samples first, so the centring means and the
    sums all pool the same points (obs gaps must not be included in the pred-only terms)."""
    valid = anom_pred.notnull() & anom_obs.notnull()
    ap = anom_pred.where(valid)
    ao = anom_obs.where(valid)
    p = ap - ap.mean(dim=dim, skipna=True)
    o = ao - ao.mean(dim=dim, skipna=True)
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


@_register("var_ratio")
def var_ratio(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    """Variability error: ratio of simulated to observed std, sigma_sim / sigma_obs.
    This is the alpha term of the NSE/KGE (Gupta et al. 2009) decomposition.
    1 = correct spread, <1 = under-dispersed, >1 = over-dispersed.
    Both fields are masked to their common valid samples so the two stds pool the same points.
    """
    valid = pred.notnull() & obs.notnull()
    p = pred.where(valid)
    o = obs.where(valid)
    return p.std(dim=dim, skipna=True) / o.std(dim=dim, skipna=True)


@_register("norm_bias")
def norm_bias(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.DataArray:
    """Unconditional (standardised) bias: (mean_sim - mean_obs) / sigma_obs.
    This is the beta term of the NSE/KGE (Gupta et al. 2009) decomposition.
    0 = unbiased; sign follows the raw bias.
    Both fields are masked to their common valid samples so the means and std pool the same points.
    """
    valid = pred.notnull() & obs.notnull()
    p = pred.where(valid)
    o = obs.where(valid)
    bias = p.mean(dim=dim, skipna=True) - o.mean(dim=dim, skipna=True)
    return bias / o.std(dim=dim, skipna=True)


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


def to_lead_hours(da: xr.DataArray) -> np.ndarray:
    """Lead time of each lead_time coordinate as integer hours."""
    return (da["lead_time"] / np.timedelta64(1, "h")).astype(int).values


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


def pool_metric(pred: xr.DataArray, obs: xr.DataArray, metric: str, dim) -> xr.DataArray:
    """Reduce one registered metric over `dim` in the squared-error domain."""
    return METRIC_REGISTRY[metric](pred, obs, dim=dim)


def compute_pooled_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str],
) -> xr.Dataset:
    """Pooled metric per lead time: one reduction over every dim except lead_time (no nesting).

    This is the climpred/xskillscore-style by-lead score -- init and space are pooled together in
    the squared-error domain, never space-RMSE-then-init-mean.
    """
    pooled_dims = [d for d in pred.dims if d != "lead_time"]
    return xr.Dataset({name: pool_metric(pred, obs, name, pooled_dims) for name in metrics})


def compute_pooled_by_day(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str],
) -> xr.Dataset:
    """Pooled metric per forecast-day bucket (24 lead-h), one reduction over init + bucket-leads + space.

    Pooling over the whole bucket (not an average of per-lead scores) keeps these consistent with
    the table_runner columns. Returns a dataset on a 'day' dim with a 'day_label' coord; empty
    buckets (beyond the horizon) are dropped.
    """
    lead_hours = to_lead_hours(pred)
    day_indices, day_labels, slices = [], [], []
    for day_idx, (low, high) in enumerate(LEAD_DAY_BUCKETS, start=1):
        lead_idx = np.where((lead_hours >= low) & (lead_hours <= high))[0]
        if lead_idx.size == 0:
            continue
        pred_bucket = pred.isel(lead_time=lead_idx)
        obs_bucket = obs.isel(lead_time=lead_idx)
        reduce_dims = list(pred_bucket.dims)
        bucket_scores = {name: pool_metric(pred_bucket, obs_bucket, name, reduce_dims) for name in metrics}
        slices.append(xr.Dataset(bucket_scores))
        day_indices.append(day_idx)
        day_labels.append(f"Day {day_idx}")

    if not slices:
        return xr.Dataset()
    ds = xr.concat(slices, dim=pd.Index(day_indices, name="day"))
    return ds.assign_coords(day_label=("day", day_labels))


def compute_daily_pooled(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metric: str,
) -> tuple[xr.DataArray, float]:
    """Daily pooled metric (one value per calendar day) and the total pooled over everything.

    Each daily value pools over space and that calendar day's timesteps in the squared-error domain;
    the total pools over all dims. For the monthly violins: per month the distribution of daily
    scores, with the total as a reference line. Returns (daily_scores, total) where daily_scores is a
    DataArray on a 'day' dim with a 'month' coord.
    """
    time_dim = next(d for d in ("valid_time", "init_time") if d in pred.dims)
    day_key = pred[time_dim].dt.floor("D").values
    unique_days = np.unique(day_key)

    values, days = [], []
    for day in unique_days:
        idx = np.where(day_key == day)[0]
        pred_day = pred.isel({time_dim: idx})
        obs_day = obs.isel({time_dim: idx})
        values.append(float(METRIC_REGISTRY[metric](pred_day, obs_day, dim=list(pred_day.dims))))
        days.append(day)

    months = pd.DatetimeIndex(days).month
    daily_scores = xr.DataArray(
        values,
        dims="day",
        coords={"day": np.array(days), "month": ("day", months)},
    )
    total = float(METRIC_REGISTRY[metric](pred, obs, dim=list(pred.dims)))
    return daily_scores, total


def daily_rmse_by_valid_day(rmse_stats: xr.Dataset, lead_idx: np.ndarray) -> tuple[xr.DataArray, float]:
    """Pooled RMSE per VALID calendar day within a lead bucket, plus the bucket total.

    Reuses the per-(init_time, lead_time) sufficient statistics (se_sum, count) from
    compute_rmse_sufficient_stats (space already reduced) — no recompute over the cube. The bucket's
    (init, lead) cells are grouped by valid day (= init + lead) and re-pooled. Returns
    (daily_rmse on a 'day' dim with a 'month' coord, total). For the per-forecast-day-bucket monthly
    violins: the month therefore reflects the valid time, not the initialisation time.
    """
    se = rmse_stats["se_sum"].isel(lead_time=lead_idx).transpose("init_time", "lead_time")
    count = rmse_stats["count"].isel(lead_time=lead_idx).transpose("init_time", "lead_time")
    init = pd.DatetimeIndex(se["init_time"].values).values
    lead = pd.TimedeltaIndex(se["lead_time"].values).values
    valid_day = pd.DatetimeIndex((init[:, None] + lead[None, :]).ravel()).floor("D")

    frame = pd.DataFrame({"day": valid_day, "se_sum": se.values.ravel(), "count": count.values.ravel().astype(float)})
    grouped = frame.groupby("day", sort=True).sum()
    grouped = grouped[grouped["count"] > 0]

    days = pd.DatetimeIndex(grouped.index)
    daily = xr.DataArray(
        np.sqrt(grouped["se_sum"].values / grouped["count"].values),
        dims="day",
        coords={"day": days.values, "month": ("day", days.month)},
    )
    total = float(np.sqrt(grouped["se_sum"].sum() / grouped["count"].sum()))
    return daily, total


def compute_rmse_sufficient_stats(
    pred: xr.DataArray,
    obs: xr.DataArray,
) -> xr.Dataset:
    """Per-(init_time, lead_time) additive sufficient statistics for a pooled-RMSE bootstrap.

    Reduces only the SPACE dims (lat/lon/pixel), keeping init_time and lead_time. Returns:
      - ``se_sum``: sum of squared error over space (NaNs skipped),
      - ``count``:  number of valid (non-NaN) pixels.
    A pooled RMSE over any set of init_times is then ``sqrt(se_sum.sum / count.sum)`` — which is
    what the block bootstrap resamples over. Keeping the sums (not per-init RMSE) is what lets the
    bootstrap re-pool correctly over resampled init_times.
    """
    se = (obs - pred) ** 2
    space_dims = [d for d in _REDUCTION_DIMS["space"] if d in se.dims]
    return xr.Dataset(
        {
            "se_sum": se.sum(dim=space_dims, skipna=True),
            "count": se.notnull().sum(dim=space_dims),
        }
    )


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


def compute_class_scores(
    pred: xr.DataArray,
    obs: xr.DataArray,
    class_coord: str,
    metrics: list[str] | None = None,
    groupby: str | None = None,
    aggregation: str = "total",
    reduction_order: tuple = ("space", "time"),
) -> xr.Dataset:
    if class_coord not in pred.coords:
        raise ValueError(f"Class coordinate '{class_coord}' not found in data coords.")

    class_bins = np.unique(pred[class_coord].values)
    class_bins = class_bins[class_bins != -1]  # Exclude invalid class bin if present
    if len(class_bins) == 0:
        # No valid class bins (e.g. anomaly path with all-NaN pixels filtering everything to -1).
        return xr.Dataset()
    slices = []
    for class_bin in class_bins:
        pred_class = pred.where(pred.coords[class_coord] == class_bin)
        obs_class = obs.where(obs.coords[class_coord] == class_bin)

        if groupby is not None:
            scores = compute_grouped_scores(
                pred_class,
                obs_class,
                metrics,
                aggregation=aggregation,
                groupby=groupby,
                reduction_order=reduction_order,
            )
        else:
            scores = compute_scores(
                pred_class, obs_class, metrics, aggregation=aggregation, reduction_order=reduction_order
            )
            dim = [d for d in _AGGREGATION_DIMS[aggregation] if d in set(pred_class.dims)] or None
            scores = scores.assign(
                {
                    "pred_mean": pred_class.mean(dim=dim, skipna=True),
                    "obs_mean": obs_class.mean(dim=dim, skipna=True),
                }
            )
        slices.append(scores)

    ds = xr.concat(slices, dim=pd.Index(class_bins, name=class_coord))
    return ds
