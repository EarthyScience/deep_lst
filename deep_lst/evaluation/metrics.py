"""Metric computation for forecast evaluation.

Two orthogonal axes organise the measures here. Axis 1, aggregation structure:
per-case-average scores (proper scoring rules / consistent scoring functions --
MAE, MSE, CRPS; Gneiting & Raftery 2007, Gneiting 2011) vs composite measures
(nonlinear functions of several averages -- RMSE, NSE, ACC, skill; Murphy &
Winkler 1987). Axis 2, forecast object: point forecasts vs distributional ones
(a probabilistic stats builder will mirror `sufficient_stats` when needed).

The evaluation chain: conditioning (masks/labels, evaluation/conditioning.py)
-> `sufficient_stats` (additive sums over the jointly valid samples)
-> `pool_stats` (regroup by summing -- the only place data is combined)
-> `measures_from_stats` (pure algebra on sums). Which tally columns a measure
needs is declared in MEASURE_REQUIRES; its tuple length tells you its family.
"""

import numpy as np
import pandas as pd
import xarray as xr

METRIC_REGISTRY: dict[str, callable] = {}

_REDUCTION_DIMS = {
    "time": ["valid_time", "init_time"],
    "space": ["lat", "lon", "pixel"],
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
    NSE ranges from -∞ to 1, where 1 indicates a perfect match between pred and obs.

    Both fields are masked to their common valid samples first, so SS_res and SS_tot pool
    the SAME points; otherwise obs-only samples (e.g. water pixels the writer NaN-masks in
    pred) enter the variance benchmark but never the error sum, biasing NSE."""
    valid = pred.notnull() & obs.notnull()
    p = pred.where(valid)
    o = obs.where(valid)
    ss_res = ((o - p) ** 2).sum(dim=dim, skipna=True)
    obs_mean = o.mean(dim=dim, skipna=True)
    ss_tot = ((o - obs_mean) ** 2).sum(dim=dim, skipna=True)
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


def local_valid_hour(da: xr.DataArray) -> xr.DataArray:
    """Local-solar hour-of-day of each valid time, per longitude.

    Local solar time = UTC valid time (init_time + lead_time) + lon/15 h. Returned as a float
    DataArray broadcasting over init_time, lead_time and whatever dim carries `lon` (the 1-D `lon`
    grid axis, or `pixel` for sparse sites). Day/night windows built on this track the diurnal
    cycle by longitude instead of a fixed UTC clock, so they stay comparable across regions.
    """
    if "valid_time" in da.coords:
        utc_hour = da["valid_time"].dt.hour  # single-lead view: the axis IS the valid time
    else:
        init_hour = da["init_time"].dt.hour
        lead_hour = (da["lead_time"] / np.timedelta64(1, "h")).astype(int)
        utc_hour = init_hour + lead_hour
    return (utc_hour + da["lon"] / 15.0) % 24


def skill_score(model_score, baseline_score):
    """RMSE-type skill, ``1 - model/baseline``. Positive means the model beats the baseline."""
    return 1.0 - model_score / baseline_score


def pool_metric(pred: xr.DataArray, obs: xr.DataArray, metric: str, dim) -> xr.DataArray:
    """Reduce one registered metric over `dim` in the squared-error domain."""
    return METRIC_REGISTRY[metric](pred, obs, dim=dim)


def _pooled_measures(pred: xr.DataArray, obs: xr.DataArray, metrics: list[str], dim) -> dict:
    """One pooled value per metric over *dim*, in the requested order.

    Chain measures (`MEASURES`) share a single sufficient-statistics pass; every other
    registered metric reduces directly via `pool_metric`.
    """
    chain_measures = tuple(name for name in metrics if name in MEASURES)
    stats = None
    if chain_measures:
        stats = sufficient_stats(pred, obs, dim=dim, extra=required_extras(chain_measures))

    pooled = {}
    for name in metrics:
        if name in MEASURES:
            pooled[name] = MEASURES[name](stats)
        else:
            pooled[name] = pool_metric(pred, obs, name, dim)
    return pooled


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
    return xr.Dataset(_pooled_measures(pred, obs, metrics, pooled_dims))


def compute_pooled_by_day(
    pred: xr.DataArray,
    obs: xr.DataArray,
    metrics: list[str],
) -> xr.Dataset:
    """Pooled metric per forecast-day bucket (24 lead-h), one reduction over init + bucket-leads + space.

    Pooling over the whole bucket (not an average of per-lead scores) keeps these consistent with
    the table_runner columns. Returns a dataset on a 'day' dim with a 'day_label' coord; empty
    buckets (beyond the horizon) are dropped. Chain measures pool via one sufficient-statistics
    pass grouped by forecast-day labels; other registered metrics reduce per bucket.
    """
    from deep_lst.evaluation.conditioning import forecast_day_labels

    chain_measures = tuple(name for name in metrics if name in MEASURES)
    registry_metrics = [name for name in metrics if name not in MEASURES]

    chain_ds = None
    if chain_measures:
        pooled_dims = [d for d in pred.dims if d != "lead_time"]
        stats = sufficient_stats(pred, obs, dim=pooled_dims, extra=required_extras(chain_measures))
        pooled = pool_stats(stats, by=forecast_day_labels(pred))
        pooled = pooled.sel(day=pooled["day"] != 0)  # leads outside every bucket
        chain_ds = measures_from_stats(pooled, chain_measures)

    registry_ds = None
    if registry_metrics:
        lead_hours = to_lead_hours(pred)
        day_indices, slices = [], []
        for day_idx, (low, high) in enumerate(LEAD_DAY_BUCKETS, start=1):
            lead_idx = np.where((lead_hours >= low) & (lead_hours <= high))[0]
            if lead_idx.size == 0:
                continue
            pred_bucket = pred.isel(lead_time=lead_idx)
            obs_bucket = obs.isel(lead_time=lead_idx)
            reduce_dims = list(pred_bucket.dims)
            bucket_scores = {name: pool_metric(pred_bucket, obs_bucket, name, reduce_dims) for name in registry_metrics}
            slices.append(xr.Dataset(bucket_scores))
            day_indices.append(day_idx)
        if slices:
            registry_ds = xr.concat(slices, dim=pd.Index(day_indices, name="day"))

    parts = {}
    for name in metrics:
        source = chain_ds if name in MEASURES else registry_ds
        if source is not None and name in source:
            parts[name] = source[name]
    if not parts:
        return xr.Dataset()
    ds = xr.Dataset(parts)
    return ds.assign_coords(day_label=("day", [f"Day {d}" for d in ds["day"].values]))


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
    space_dims = [d for d in _REDUCTION_DIMS["space"] if d in pred.dims]
    stats = sufficient_stats(pred, obs, dim=space_dims)
    return xr.Dataset({"se_sum": stats["se_sum"], "count": stats["n"]})


_EXTRA_STATS = ("ae_sum", "o_sum", "o2_sum", "p_sum", "p2_sum", "po_sum")


def sufficient_stats(pred: xr.DataArray, obs: xr.DataArray, dim, extra: tuple = ()) -> xr.Dataset:
    """Additive sufficient statistics of the forecast-observation pairs, reduced over *dim*.

    Always returns ``n``, ``e_sum``, ``se_sum`` (error convention ``e = pred - obs``);
    each name in *extra* adds a variable (currently ``ae_sum``). All variables are sums
    or counts over the jointly valid samples, so any later regrouping (`pool_stats`) is
    exact. Choose *dim* as the dims that will never be conditioned on downstream --
    reducing a dim is the one irreversible act in the chain.
    """
    unknown = set(extra) - set(_EXTRA_STATS)
    if unknown:
        raise ValueError(f"Unknown extra statistics: {sorted(unknown)}")
    valid = pred.notnull() & obs.notnull()
    signed_error = pred - obs  # joint-NaN by construction

    stats = {
        "n": valid.sum(dim=dim),
        "e_sum": signed_error.sum(dim=dim, skipna=True),
        "se_sum": (signed_error**2).sum(dim=dim, skipna=True),
    }
    if "ae_sum" in extra:
        stats["ae_sum"] = np.abs(signed_error).sum(dim=dim, skipna=True)
    if "o_sum" in extra or "o2_sum" in extra:
        # float64: the NSE finish cancels o2_sum against o_sum²/n, which float32 sums
        # of ~300 K values cannot survive.
        obs_valid = obs.where(valid).astype("float64")
        if "o_sum" in extra:
            stats["o_sum"] = obs_valid.sum(dim=dim, skipna=True)
        if "o2_sum" in extra:
            stats["o2_sum"] = (obs_valid**2).sum(dim=dim, skipna=True)
    if "p_sum" in extra or "p2_sum" in extra or "po_sum" in extra:
        pred_valid = pred.where(valid).astype("float64")
        if "p_sum" in extra:
            stats["p_sum"] = pred_valid.sum(dim=dim, skipna=True)
        if "p2_sum" in extra:
            stats["p2_sum"] = (pred_valid**2).sum(dim=dim, skipna=True)
        if "po_sum" in extra:
            obs_valid = obs.where(valid).astype("float64")
            stats["po_sum"] = (pred_valid * obs_valid).sum(dim=dim, skipna=True)
    return xr.Dataset(stats)


def pool_stats(stats: xr.Dataset, dim=None, by: xr.DataArray | None = None) -> xr.Dataset:
    """Sum additive statistics over *dim*; then, if *by* is given, group-by-sum over its labels.

    *by* is a label DataArray on one or more of the remaining dims; the output gains one
    dim named after ``by.name`` with one entry per unique label. Multi-dim labels (e.g.
    valid calendar day on ``(init_time, lead_time)``) are handled by stacking the label
    dims first, with the labels riding along as a coordinate so alignment is exact.
    """
    if dim is not None:
        stats = stats.sum(dim=dim)
    if by is None:
        return stats
    if by.ndim == 1:
        return stats.groupby(by).sum(dim=by.dims[0])
    stacked = stats.assign_coords({by.name: by}).stack(_cell=by.dims)
    return stacked.groupby(by.name).sum(dim="_cell")


def _measure_rmse(stats: xr.Dataset) -> xr.DataArray:
    return np.sqrt(stats["se_sum"] / stats["n"])


def _measure_bias(stats: xr.Dataset) -> xr.DataArray:
    return stats["e_sum"] / stats["n"]


def _measure_mae(stats: xr.Dataset) -> xr.DataArray:
    return stats["ae_sum"] / stats["n"]


def _measure_nse(stats: xr.Dataset) -> xr.DataArray:
    ss_tot = stats["o2_sum"] - stats["o_sum"] ** 2 / stats["n"]
    return 1.0 - stats["se_sum"] / ss_tot


MEASURES: dict[str, callable] = {
    "rmse": _measure_rmse,
    "bias": _measure_bias,
    "mae": _measure_mae,
    "nse": _measure_nse,
}


def _measure_acc(stats: xr.Dataset) -> xr.DataArray:
    return stats["po_sum"] / np.sqrt(stats["p2_sum"] * stats["o2_sum"])


def _measure_activity(stats: xr.Dataset) -> xr.DataArray:
    return np.sqrt(stats["p2_sum"] / stats["n"] - (stats["p_sum"] / stats["n"]) ** 2)


# Anomaly-input measures: same finish machinery, but the stats must have been built
# from anomaly fields (pred - clim, obs - clim). Kept separate from MEASURES so the
# legacy pooled entry points keep routing acc/activity through METRIC_REGISTRY
# bit-identically; the products engine consults both dicts.
ANOMALY_MEASURES: dict[str, callable] = {"acc": _measure_acc, "activity": _measure_activity}

MEASURE_REQUIRES = {
    "rmse": ("se_sum", "n"),
    "bias": ("e_sum", "n"),
    "mae": ("ae_sum", "n"),
    "nse": ("se_sum", "o_sum", "o2_sum", "n"),
    "acc": ("po_sum", "p2_sum", "o2_sum", "n"),
    "activity": ("p_sum", "p2_sum", "n"),
}


def required_extras(measures: tuple) -> tuple:
    """The `extra` names `sufficient_stats` must accumulate for these measures."""
    base = {"n", "e_sum", "se_sum"}
    needed = {name for measure in measures for name in MEASURE_REQUIRES[measure]}
    return tuple(sorted(needed - base))


def measures_from_stats(stats: xr.Dataset, measures: tuple) -> xr.Dataset:
    """Evaluate verification measures from pooled sufficient statistics.

    Pure algebra on sums -- no reductions happen here. An all-masked subset yields
    ``n = 0`` and therefore NaN, which is the intended "no data" signal. Anomaly
    measures (acc, activity) resolve too, but expect anomaly-built stats.
    """
    finished = {}
    for name in measures:
        finisher = MEASURES.get(name) or ANOMALY_MEASURES.get(name)
        if finisher is None:
            raise KeyError(f"No finisher for measure {name!r}")
        finished[name] = finisher(stats)
    return xr.Dataset(finished)
