"""Diurnal cycle plots for forecast evaluation.

diurnal_line(diurnal_by_case, ...)
    Line plot of a metric by hour of day, one line per case (A, B, C).
    Shading shows ± 1 std of individual errors per hour.
    Input: dict of xr.Datasets with dim ``hour``, output of
    ``compute_diurnal_scores()``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray import Dataset

from deep_lst.evaluation.plots._design import (
    _LABEL_ALIASES,
    BASELINE_LINESTYLE,
    CASE_COLORS,
    CASE_LINESTYLES,
    DEFAULT_COLOR,
    OBS_COLOR,
    OBS_LINESTYLE,
    RC,
)

_DEFAULT_COLOR = DEFAULT_COLOR
_PRED_COLOR = CASE_COLORS.get("MLP-Rollout", "#d62728")
_OBS_COLOR = OBS_COLOR


def diurnal_line(
    diurnal_by_case: dict[str, xr.Dataset],
    metric: str = "rmse",
    save_path: Path = None,
    title: str = None,
    ylabel: str = None,
    show_std: bool = True,
) -> plt.Figure:
    """Line plot of a diurnal metric for multiple forecast cases.

    Parameters
    ----------
    diurnal_by_case : dict[str, xr.Dataset]
        ``{case_label: diurnal_dataset}`` — output of
        ``compute_diurnal_scores()``.  Each dataset has dim ``hour`` (0–23)
        and variables for each metric plus ``std`` and ``n_valid``.
    metric : str
        Which variable to plot on y-axis. Default: "rmse".
    save_path : Path, optional
        If given, save the figure here.
    title : str, optional
        Figure title.
    ylabel : str, optional
        Y-axis label. Defaults to metric name in uppercase.
    show_std : bool
        If True, draw ± 1 std shading around each line. Default: True.

    Returns
    -------
    plt.Figure
    """
    plt.rcParams.update(RC)

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for case, ds in diurnal_by_case.items():
        q50_key = f"{metric}_q50"
        if metric not in ds and q50_key not in ds:
            print(f"Warning: metric '{metric}' not in dataset for case '{case}'. Skipping.")
            continue

        color = CASE_COLORS.get(case, _DEFAULT_COLOR)
        vals = ds[q50_key if q50_key in ds else metric].values.copy()
        vals[vals == 0] = np.nan  # hours with zero valid samples → NaN gap

        ls = CASE_LINESTYLES.get(case, BASELINE_LINESTYLE)
        ax.plot(hours, vals, color=color, linestyle=ls, linewidth=2, label=case, marker="o", markersize=4)

        q05_key, q95_key = f"{metric}_q05", f"{metric}_q95"
        if show_std and q05_key in ds and q95_key in ds:
            lo = ds[q05_key].values.copy()
            hi = ds[q95_key].values.copy()
            lo[~np.isfinite(vals)] = np.nan
            hi[~np.isfinite(vals)] = np.nan
            ax.fill_between(hours, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(ylabel or metric.upper() + " (K)")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


_BASELINE_COLORS = CASE_COLORS  # baselines share the same central palette


def diurnal_mean_plot(
    diurnal_mean_ds: Dataset,
    var: str = "LST",
    baselines_mean: dict[str, np.ndarray] | None = None,
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
    case_label: str | None = None,
) -> plt.Figure:
    """Mean LST by hour of day (pred vs obs), averaged over all pixels and days.

    Parameters
    ----------
    diurnal_mean_ds : xr.Dataset
        Output of compute_diurnal_mean(). Must contain pred_mean, obs_mean,
        pred_std, obs_std with dim hour.
    var : str
        Variable label for axis.
    baselines_mean : dict[str, np.ndarray], optional
        Per-baseline hourly means, each shape (24,), keyed by display label
        (e.g. "ERA5", "hourly clim"). Each plotted as a dashed reference line.
    save_path : Path, optional
    title : str, optional
    eval_region : str
        Used in default title.
    """
    plt.rcParams.update(RC)

    hours = np.arange(24)
    pred_mean = diurnal_mean_ds["pred_mean_q50" if "pred_mean_q50" in diurnal_mean_ds else "pred_mean"].values
    obs_mean = diurnal_mean_ds["obs_mean_q50" if "obs_mean_q50" in diurnal_mean_ds else "obs_mean"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))

    pred_color = CASE_COLORS.get(case_label, _PRED_COLOR) if case_label else _PRED_COLOR
    ax.plot(hours, obs_mean, color=_OBS_COLOR, linewidth=2, marker="o", markersize=4, label="SEVIRI LST")
    ax.plot(hours, pred_mean, color=pred_color, linewidth=2, marker="o", markersize=4, label="Deep LST", alpha=0.85)
    if "obs_mean_q05" in diurnal_mean_ds:
        ax.fill_between(
            hours,
            diurnal_mean_ds["obs_mean_q05"].values,
            diurnal_mean_ds["obs_mean_q95"].values,
            color=_OBS_COLOR,
            alpha=0.15,
        )
    if "pred_mean_q05" in diurnal_mean_ds:
        ax.fill_between(
            hours,
            diurnal_mean_ds["pred_mean_q05"].values,
            diurnal_mean_ds["pred_mean_q95"].values,
            color=pred_color,
            alpha=0.15,
        )

    for label, bl_mean in (baselines_mean or {}).items():
        color = _BASELINE_COLORS.get(label, _DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(label, BASELINE_LINESTYLE)
        ax.plot(hours, bl_mean, color=color, linestyle=ls, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(f"Mean {var} (K)")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")
    return fig


def diurnal_delta_plot(
    diurnal_mean_ds: Dataset,
    var: str = "LST",
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
) -> plt.Figure:
    """Hour-to-hour temperature change (ΔT/h) by hour of day, pred vs obs.

    delta[h] = mean_LST[h+1] - mean_LST[h] — shows how quickly LST
    heats up or cools down at each point in the diurnal cycle.

    Parameters
    ----------
    diurnal_mean_ds : xr.Dataset
        Output of compute_diurnal_mean(). Must contain pred_delta, obs_delta.
    """
    plt.rcParams.update(RC)

    hours = np.arange(24)
    pred_delta = diurnal_mean_ds["pred_delta"].values
    obs_delta = diurnal_mean_ds["obs_delta"].values

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(hours, obs_delta, color=_OBS_COLOR, linewidth=2, marker="o", markersize=4, label="SEVIRI LST")
    ax.plot(hours, pred_delta, color=_PRED_COLOR, linewidth=2, marker="o", markersize=4, label="Deep LST", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(f"Δ{var} (K/h)")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title or f"{eval_region} — diurnal {var} change (ΔT/h)")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")
    return fig


def diurnal_delta_error_plot(
    model_diurnal_mean: Dataset,
    baselines_diurnal_mean: dict[str, Dataset],
    var: str = "LST",
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
    case_label: str | None = None,
    lead_time: str | None = None,
) -> plt.Figure:
    """Error in ΔT/h by hour of day: model and all baselines vs observed.

    Plots ``pred_delta − obs_delta`` for each case (model + baselines).
    A flat zero line indicates perfect reproduction of the diurnal heating rate.

    Parameters
    ----------
    model_diurnal_mean : xr.Dataset
        Output of ``compute_diurnal_mean(pred, obs)`` — contains ``pred_delta``
        (model) and ``obs_delta`` (observation, same for all cases).
    baselines_diurnal_mean : dict[str, xr.Dataset]
        ``{label: compute_diurnal_mean(baseline_da, obs)}`` — one entry per
        baseline. ``pred_delta`` here is the baseline's Δ.
    case_label : str, optional
        Display label for the model (colour looked up from CASE_COLORS).
    """
    plt.rcParams.update(RC)

    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.8)

    # Model
    pred_label = "Deep LST"
    pred_color = CASE_COLORS.get(pred_label, _PRED_COLOR)
    ax.plot(
        hours,
        model_diurnal_mean["delta_err_q50"].values,
        color=pred_color,
        linewidth=2,
        marker="o",
        markersize=4,
        label=pred_label,
        zorder=3,
    )
    ax.fill_between(
        hours,
        model_diurnal_mean["delta_err_q05"].values,
        model_diurnal_mean["delta_err_q95"].values,
        color=pred_color,
        alpha=0.2,
    )

    # Baselines
    for label, ds in baselines_diurnal_mean.items():
        color = CASE_COLORS.get(label, _DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(label, BASELINE_LINESTYLE)
        display_label = _LABEL_ALIASES.get(label, label)
        ax.plot(
            hours,
            ds["delta_err_q50"].values,
            color=color,
            linestyle=ls,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=display_label,
            alpha=0.85,
        )
        ax.fill_between(hours, ds["delta_err_q05"].values, ds["delta_err_q95"].values, color=color, alpha=0.15)

    ax.set_xlabel("Hour of day (UTC)", fontweight="bold")
    ax.set_ylabel(f"Δ{var} bias (K/h)", fontweight="bold")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    if lead_time is not None:
        ax.text(0.02, 0.97, f"Lead time = {lead_time}", transform=ax.transAxes, va="top", ha="left", fontsize=12)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"Saved: {save_path}")
    return fig


def diurnal_delta_comparison(
    diurnal_mean_by_case: dict[str, Dataset],
    var: str = "LST",
    save_path: Path = None,
    title: str = None,
    eval_region: str = "",
) -> plt.Figure:
    """ΔT/h by hour of day comparing multiple forecast cases against observed.

    Plots one observed line (averaged across all cases) plus one predicted
    line per case (e.g. MLP-Diag and MLP-Rollout).

    Parameters
    ----------
    diurnal_mean_by_case : dict[str, xr.Dataset]
        ``{case_label: diurnal_mean_dataset}`` — output of
        ``compute_diurnal_mean()``. Each dataset has dim ``hour`` and
        variables ``pred_delta`` and ``obs_delta``.
    """
    plt.rcParams.update(RC)

    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    # Observed: average obs_delta across cases (should be nearly identical)
    obs_deltas = [ds["obs_delta"].values for ds in diurnal_mean_by_case.values()]
    obs_mean = np.nanmean(obs_deltas, axis=0)
    ax.plot(
        hours,
        obs_mean,
        color=_OBS_COLOR,
        linestyle=OBS_LINESTYLE,
        linewidth=2,
        marker="o",
        markersize=4,
        label="SEVIRI obs",
        zorder=3,
    )

    # One predicted line per case
    for case, ds in diurnal_mean_by_case.items():
        color = CASE_COLORS.get(case, _DEFAULT_COLOR)
        ls = CASE_LINESTYLES.get(case, BASELINE_LINESTYLE)
        pred_delta = ds["pred_delta"].values.copy()
        ax.plot(
            hours, pred_delta, color=color, linestyle=ls, linewidth=2, marker="o", markersize=4, label=case, alpha=0.85
        )

    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel(f"Δ{var} (K/h)")
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlim(-0.5, 23.5)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title or f"{eval_region} — diurnal {var} change comparison (ΔT/h)")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")
    return fig
