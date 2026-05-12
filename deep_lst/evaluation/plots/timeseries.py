"""Time-series plot for forecast evaluation.

timeseries_plot(ds, var, lead_time, ...)
    Plots region-mean pred and obs as a continuous time series over the full
    test period at a given lead time.  Useful for a sanity-check of whether
    the model tracks the diurnal cycle and multi-day variation.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from deep_lst.evaluation.metrics import to_valid_time
from deep_lst.evaluation.plots._design import (
    BASELINE_LINESTYLE,
    CASE_COLORS,
    CASE_LINESTYLES,
    DEFAULT_COLOR,
    OBS_COLOR,
    RC,
)

PRED_COLOR = CASE_COLORS.get("MLP-Rollout", "#d62728")
ERA5_COLOR = CASE_COLORS.get("ERA5", "#1f77b4")

_BASELINE_COLORS = CASE_COLORS
_BASELINE_DEFAULT_COLOR = DEFAULT_COLOR


def timeseries_plot(
    ds: xr.Dataset,
    var: str = "LST",
    lead_time: str = "1h",
    baselines: dict[str, xr.DataArray] | None = None,
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
    show_std: bool = True,
    case_label: str | None = None,
) -> plt.Figure:
    """Region-mean time series of pred vs obs at a fixed lead time.

    Aggregation
    -----------
    - Selects the requested lead_time from the (init_time, lead_time) forecast.
    - Converts to valid_time coordinates (valid_time = init_time + lead_time).
    - Takes the nanmean over all lat/lon pixels → one value per timestep.
    - If show_std=True, adds ± 1 spatial std shading (variability across pixels).

    Parameters
    ----------
    ds : xr.Dataset
        ForecastResult dataset with dims (init_time, lead_time, lat, lon).
    var : str
        Variable name, e.g. "LST".
    lead_time : str
        Lead time to select, e.g. "1h" or "3h".
    baselines : dict[str, xr.DataArray], optional
        Aligned baseline DataArrays keyed by display label (e.g. "ERA5",
        "hourly clim"). Each plotted as a dashed reference line.
    save_path : Path, optional
    title : str, optional
    eval_region : str
        Used in default title.
    show_std : bool
        If True, draw ± 1 spatial std shading around pred and obs means.
    """
    plt.rcParams.update(RC)

    pred, obs = to_valid_time(ds, var, lead_time=lead_time)

    # Spatial mean and std (over all non-time dims: lat/lon for dense, pixel for sparse)
    spatial_dims = [d for d in pred.dims if d != "valid_time"]
    pred_mean = pred.mean(dim=spatial_dims, skipna=True)
    obs_mean = obs.mean(dim=spatial_dims, skipna=True)
    times = pd.DatetimeIndex(pred_mean["valid_time"].values)

    fig, ax = plt.subplots(figsize=(14, 4))

    _pred_color = CASE_COLORS.get(case_label, PRED_COLOR) if case_label else PRED_COLOR
    _pred_label = f"{case_label} (lt={lead_time})" if case_label else f"Model pred (lt={lead_time})"
    ax.plot(
        times, obs_mean.values, color=OBS_COLOR, linewidth=1.0, marker=".", markersize=2, label="SEVIRI obs", zorder=3
    )
    ax.plot(
        times,
        pred_mean.values,
        color=_pred_color,
        linewidth=1.0,
        marker=".",
        markersize=2,
        label=_pred_label,
        zorder=4,
        alpha=0.85,
    )

    if show_std:
        pred_std = pred.std(dim=spatial_dims, skipna=True)
        obs_std = obs.std(dim=spatial_dims, skipna=True)
        ax.fill_between(times, obs_mean - obs_std, obs_mean + obs_std, color=OBS_COLOR, alpha=0.15, zorder=2)
        ax.fill_between(times, pred_mean - pred_std, pred_mean + pred_std, color=_pred_color, alpha=0.15, zorder=2)

    for label, da in (baselines or {}).items():
        bl_aligned = da.reindex(valid_time=pred["valid_time"])
        spatial_dims = [d for d in bl_aligned.dims if d != "valid_time"]
        bl_mean = bl_aligned.mean(dim=spatial_dims, skipna=True) if spatial_dims else bl_aligned
        color = _BASELINE_COLORS.get(label, _BASELINE_DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(label, BASELINE_LINESTYLE)
        ax.plot(times, bl_mean.values, color=color, linestyle=ls, linewidth=1.0, label=label, zorder=2, alpha=0.75)

    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel(f"{var} (K)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title is None:
        title = f"{eval_region} — region-mean {var}, lead_time={lead_time}"
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig
