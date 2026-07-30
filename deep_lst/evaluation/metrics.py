import numpy as np
import pandas as pd
import xarray as xr

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


MEASURE_REQUIRES = {
    "rmse": ("se_sum", "n"),
    "bias": ("e_sum", "n"),
    "mae": ("ae_sum", "n"),
    "nse": ("se_sum", "o_sum", "o2_sum", "n"),
    "acc": ("po_sum", "p2_sum", "o2_sum", "n"),
    "activity": ("p_sum", "p2_sum", "n"),
    "rmse_debiased": ("se_sum", "e_sum", "n"),
    "var_ratio": ("p_sum", "p2_sum", "o_sum", "o2_sum", "n"),
    "norm_bias": ("p_sum", "o_sum", "o2_sum", "n"),
    "c_acc": ("po_sum", "p_sum", "p2_sum", "o_sum", "o2_sum", "n"),
}


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


def sufficient_stats(pred: xr.DataArray, obs: xr.DataArray, dim) -> xr.Dataset:
    valid = pred.notnull() & obs.notnull()
    signed_error = pred - obs  # joint-NaN by construction
    obs_valid = obs.where(valid).astype("float64")
    pred_valid = pred.where(valid).astype("float64")

    return xr.Dataset(
        {
            "n": valid.sum(dim=dim),
            "e_sum": signed_error.sum(dim=dim, skipna=True),
            "se_sum": (signed_error**2).sum(dim=dim, skipna=True),
            "ae_sum": np.abs(signed_error).sum(dim=dim, skipna=True),
            "o_sum": obs_valid.sum(dim=dim, skipna=True),
            "o2_sum": (obs_valid**2).sum(dim=dim, skipna=True),
            "p_sum": pred_valid.sum(dim=dim, skipna=True),
            "p2_sum": (pred_valid**2).sum(dim=dim, skipna=True),
            "po_sum": (pred_valid * obs_valid).sum(dim=dim, skipna=True),
        }
    )


def pool_stats(stats: xr.Dataset, dim=None, by: xr.DataArray | None = None) -> xr.Dataset:
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
    # ss_tot on the SAME jointly valid samples as ss_res -- obs-only samples (e.g. water
    # pixels NaN-masked in pred) must not enter the variance benchmark.
    ss_tot = stats["o2_sum"] - stats["o_sum"] ** 2 / stats["n"]
    return 1.0 - stats["se_sum"] / ss_tot


def _measure_rmse_debiased(stats: xr.Dataset) -> xr.DataArray:
    # per-cell mean error removed before pooling: sqrt((se - e^2/n) / n), clipped for
    # float cancellation; the granularity of the debiasing is the tally's granularity
    ss_debiased = (stats["se_sum"] - stats["e_sum"] ** 2 / stats["n"]).clip(min=0.0)
    return np.sqrt(ss_debiased / stats["n"])


def _measure_var_ratio(stats: xr.Dataset) -> xr.DataArray:
    # sigma_sim / sigma_obs (population stds, jointly valid samples) -- KGE alpha term
    var_pred = stats["p2_sum"] / stats["n"] - (stats["p_sum"] / stats["n"]) ** 2
    var_obs = stats["o2_sum"] / stats["n"] - (stats["o_sum"] / stats["n"]) ** 2
    return np.sqrt(var_pred / var_obs)


def _measure_norm_bias(stats: xr.Dataset) -> xr.DataArray:
    # (mean_sim - mean_obs) / sigma_obs -- KGE beta term
    var_obs = stats["o2_sum"] / stats["n"] - (stats["o_sum"] / stats["n"]) ** 2
    return (stats["p_sum"] / stats["n"] - stats["o_sum"] / stats["n"]) / np.sqrt(var_obs)


def _measure_c_acc(stats: xr.Dataset) -> xr.DataArray:
    # Pearson r (centred correlation) from the cross-moments, jointly valid samples
    n = stats["n"]
    cov = stats["po_sum"] - stats["p_sum"] * stats["o_sum"] / n
    var_pred = stats["p2_sum"] - stats["p_sum"] ** 2 / n
    var_obs = stats["o2_sum"] - stats["o_sum"] ** 2 / n
    return cov / np.sqrt(var_pred * var_obs)


def _measure_acc(stats: xr.Dataset) -> xr.DataArray:
    return stats["po_sum"] / np.sqrt(stats["p2_sum"] * stats["o2_sum"])


def _measure_activity(stats: xr.Dataset) -> xr.DataArray:
    return np.sqrt(stats["p2_sum"] / stats["n"] - (stats["p_sum"] / stats["n"]) ** 2)


MEASURES: dict[str, callable] = {
    "rmse": _measure_rmse,
    "bias": _measure_bias,
    "mae": _measure_mae,
    "nse": _measure_nse,
    "rmse_debiased": _measure_rmse_debiased,
    "var_ratio": _measure_var_ratio,
    "norm_bias": _measure_norm_bias,
    "c_acc": _measure_c_acc,
}

ANOMALY_MEASURES: dict[str, callable] = {"acc": _measure_acc, "activity": _measure_activity}


def measures_from_stats(stats: xr.Dataset, measures: tuple) -> xr.Dataset:
    finished = {}
    for name in measures:
        finisher = MEASURES.get(name) or ANOMALY_MEASURES.get(name)
        if finisher is None:
            raise KeyError(f"No finisher for measure {name!r}")
        finished[name] = finisher(stats)
    return xr.Dataset(finished)
