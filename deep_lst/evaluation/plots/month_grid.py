"""Generic two-panel (Day | Night) months x columns heatmap.

Shared renderer for the forecast-limit and monthly-score heatmaps: rows = months (ascending),
columns = regions or classes, one panel per day/night. The caller supplies the value column, the
colormap/norm and colorbar, plus optional latitude top-axis labels and a train/eval separator.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize

from deep_lst.evaluation.plots._design import RC

MONTHS = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _latitude_labels(df, col_order, training_label):
    """Top-axis tick labels: centre latitude per column, blank for the training column."""
    latitude = df.groupby("region")["latitude"].first()
    return ["" if name == training_label else f"{latitude[name]:.1f}°" for name in col_order]


# Toggle for all heatmap colorbars: True -> discrete `step`-K blocks; False -> continuous.
DISCRETE_COLORBARS = False


def binned_norm(vmin, vmax, step=0.5, cmap_name="viridis"):
    """Sequential colormap + norm for the heatmaps; `step`-K discrete blocks or continuous.

    Returns (cmap, norm, boundaries). Boundaries snap outward to whole `step` multiples (kept as the
    colorbar tick positions in both modes). Missing cells are grey. Flip DISCRETE_COLORBARS to switch.
    """
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    boundaries = np.arange(lo, hi + step / 2, step)
    if DISCRETE_COLORBARS:
        cmap = plt.get_cmap(cmap_name, len(boundaries) - 1).copy()
        cmap.set_bad("lightgrey")
        return cmap, BoundaryNorm(boundaries, cmap.N), boundaries
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")
    return cmap, Normalize(lo, hi), boundaries


def _cell_matrix(df, col_order, kind, value_col):
    """months x columns matrix of `value_col` for one day/night panel."""
    panel = df[df["daynight"] == kind]
    lookup = {(row["region"], row["month"]): row[value_col] for _, row in panel.iterrows()}
    matrix = np.full((len(MONTHS), len(col_order)), np.nan)
    for col, name in enumerate(col_order):
        for row, month in enumerate(MONTHS):
            matrix[row, col] = lookup.get((name, month), np.nan)
    return matrix


def month_grid_heatmap(
    df,
    col_order,
    value_col,
    save_path,
    *,
    cmap,
    norm,
    cbar_label,
    cbar_ticks=None,
    cbar_ticklabels=None,
    top_labels=None,
    separator_x=None,
    col_gridlines=False,
    figsize_per_col=1.2,
    figsize_base=2.0,
):
    """Render the two-panel months x columns heatmap of `value_col`.

    top_labels (e.g. centre latitudes) draw a secondary top axis; separator_x draws a vertical rule
    (train | eval); col_gridlines draws dashed rules between columns. Missing cells use cmap.set_bad.
    """
    plt.rcParams.update(RC)

    fig, axes = plt.subplots(1, 2, figsize=(figsize_per_col * len(col_order) + figsize_base, 6), sharey=True)
    image = None
    for ax, kind in zip(axes, ("day", "night")):
        matrix = _cell_matrix(df, col_order, kind, value_col)
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, rotation=45, ha="right", fontweight="bold")
        ax.set_title(kind.capitalize(), pad=36 if top_labels is not None else 6, fontweight="bold")

        if separator_x is not None:
            ax.axvline(separator_x, color="gray", linestyle="-", linewidth=1.5)
        if col_gridlines:
            for boundary in np.arange(0.5, len(col_order) - 0.5, 1.0):
                ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.5)
        if top_labels is not None:
            top = ax.secondary_xaxis("top")
            top.set_xticks(range(len(col_order)))
            top.set_xticklabels(top_labels, rotation=45, ha="left")

    axes[0].set_yticks(range(len(MONTHS)))
    axes[0].set_yticklabels(MONTH_LABELS, fontweight="bold")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04, ticks=cbar_ticks)
    colorbar.set_label(cbar_label, fontweight="bold")
    if cbar_ticklabels is not None:
        colorbar.set_ticklabels(cbar_ticklabels)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")


def monthly_rmse_heatmap(
    df,
    col_order,
    save_path,
    *,
    value_col="rmse",
    step=1.0,
    value_range=None,
    cbar_label="RMSE [K]",
    top_labels=None,
    separator_x=None,
    col_gridlines=False,
    figsize_per_col=1.2,
    figsize_base=2.0,
):
    """Monthly RMSE heatmap: discretised viridis in `step`-K blocks.

    Autoscaled to this df by default; pass `value_range=(lo, hi)` to fix the scale (e.g. to share one
    colorbar between the raw and debiased region heatmaps). Colorbar blocks are `step` K wide.
    """
    if value_range is None:
        values = df[value_col].to_numpy(dtype=float)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    else:
        lo, hi = value_range
    cmap, norm, boundaries = binned_norm(lo, hi, step=step)
    ticks = boundaries
    month_grid_heatmap(
        df,
        col_order,
        value_col,
        save_path,
        cmap=cmap,
        norm=norm,
        cbar_label=cbar_label,
        cbar_ticks=ticks,
        cbar_ticklabels=[f"{tick:.1f}" for tick in ticks],
        top_labels=top_labels,
        separator_x=separator_x,
        col_gridlines=col_gridlines,
        figsize_per_col=figsize_per_col,
        figsize_base=figsize_base,
    )


# --- month x hour RMSE "fingerprint", one figure per region (one panel per lead window) -----------
def _fingerprint_panel(fig, ax, matrix, cmap, norm, bounds, title, cbar_label):
    """Draw one month (x) x hour (y) heatmap panel with its colorbar."""
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LABELS, rotation=45, ha="right", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=bounds)
    colorbar.set_label(cbar_label, fontweight="bold")


def _hour_yaxis(ax):
    ax.set_yticks(range(0, 24, 3))
    ax.set_yticklabels([f"{hour:02d}" for hour in range(0, 24, 3)])
    ax.set_ylabel("Hour of day (UTC)", fontweight="bold")


def fingerprint_rmse_heatmap(panels, save_path, *, step=1.0):
    """Per-region RMSE fingerprint: month (x) x hour-of-day (y), one panel per lead window.

    panels: list of (title, matrix) where matrix is (24 hour, 12 month) RMSE [K]. Each panel is
    discretised in `step`-K viridis blocks, autoscaled to its own data with its own colorbar.
    """
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels) + 1.0, 6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, matrix) in zip(axes, panels):
        cmap, norm, bounds = binned_norm(float(np.nanmin(matrix)), float(np.nanmax(matrix)), step=step)
        _fingerprint_panel(fig, ax, matrix, cmap, norm, bounds, title, "RMSE [K]")
    _hour_yaxis(axes[0])

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")


def fingerprint_decomposition_heatmap(raw, centered, collapse, title, save_path, *, step=1.0):
    """Bias decomposition fingerprint: RMSE | RMSE debiased | % MSE collapsed (month x hour).

    raw/centered/collapse are (24 hour, 12 month) matrices. raw and centered share one K colour scale
    so the collapse from per-pixel debiasing is visible; the collapse panel is its own 0-100 % scale.
    """
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    both = np.concatenate([raw.ravel(), centered.ravel()])
    k_cmap, k_norm, k_bounds = binned_norm(float(np.nanmin(both)), float(np.nanmax(both)), step=step)
    _fingerprint_panel(fig, axes[0], raw, k_cmap, k_norm, k_bounds, "RMSE [K]", "RMSE [K]")
    _fingerprint_panel(fig, axes[1], centered, k_cmap, k_norm, k_bounds, "RMSE debiased [K]", "RMSE [K]")

    c_cmap, c_norm, c_bounds = binned_norm(0.0, 100.0, step=10.0, cmap_name="magma")
    _fingerprint_panel(fig, axes[2], collapse, c_cmap, c_norm, c_bounds, "% MSE collapsed", "% MSE collapsed")

    _hour_yaxis(axes[0])
    fig.suptitle(title, fontweight="bold")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")


def fingerprint_decomposition_honest_heatmap(
    raw, centered, collapse, median_n, honest_collapse, corrected_rmse, title, save_path, *, step=1.0
):
    """Six-panel honest decomposition with paired colorbars and an explicit difference map.

    Row 1: RMSE | debiased RMSE (Bessel) [SHARED K scale] | Delta debiased = in-sample - Bessel [K,
    diverging at 0]. Row 2: median n_p (viridis, log) | % collapsed (in-sample) | honest % collapsed
    (Bessel) [SHARED % scale]. The near-zero Delta panel shows the Bessel correction barely moves the
    debiased RMSE where pixels are well sampled. All matrices are (24 hour, 12 month).
    """
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 3, figsize=(14, 11), sharex=True)

    # Row 1: raw vs Bessel-debiased share ONE K scale; the difference gets its own diverging scale.
    both_k = np.concatenate([raw.ravel(), corrected_rmse.ravel()])
    k_cmap, k_norm, k_bounds = binned_norm(float(np.nanmin(both_k)), float(np.nanmax(both_k)), step=step)
    _fingerprint_panel(fig, axes[0, 0], raw, k_cmap, k_norm, k_bounds, "RMSE [K]", "RMSE [K]")
    _fingerprint_panel(
        fig, axes[0, 1], corrected_rmse, k_cmap, k_norm, k_bounds, "RMSE debiased (Bessel) [K]", "RMSE [K]"
    )

    diff = centered - corrected_rmse
    absmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
    d_cmap = plt.get_cmap("RdBu_r").copy()
    d_cmap.set_bad("lightgrey")
    _fingerprint_panel(
        fig,
        axes[0, 2],
        diff,
        d_cmap,
        Normalize(-max(absmax, 1e-6), max(absmax, 1e-6)),
        None,
        "Δ debiased: in-sample − Bessel [K]",
        "Δ [K]",
    )

    # Row 2: median n_p on a log viridis scale; the two collapse panels share ONE % scale.
    n_hi = float(np.nanmax(median_n)) if np.isfinite(median_n).any() else 2.0
    n_cmap = plt.get_cmap("viridis").copy()
    n_cmap.set_bad("lightgrey")
    _fingerprint_panel(
        fig, axes[1, 0], median_n, n_cmap, LogNorm(1.0, max(n_hi, 2.0)), None, "median n_p per pixel (log)", "n_p"
    )

    both_c = np.concatenate([collapse.ravel(), honest_collapse.ravel()])
    c_lo = float(np.nanmin(both_c)) if np.isfinite(both_c).any() else 0.0
    c_hi = float(np.nanmax(both_c)) if np.isfinite(both_c).any() else 100.0
    c_cmap, c_norm, c_bounds = binned_norm(min(c_lo, 0.0), max(c_hi, 10.0), step=10.0, cmap_name="magma")
    _fingerprint_panel(fig, axes[1, 1], collapse, c_cmap, c_norm, c_bounds, "% MSE collapsed (in-sample)", "%")
    _fingerprint_panel(
        fig, axes[1, 2], honest_collapse, c_cmap, c_norm, c_bounds, "honest % MSE collapsed (Bessel)", "%"
    )

    _hour_yaxis(axes[0, 0])
    _hour_yaxis(axes[1, 0])
    fig.suptitle(title, fontweight="bold")

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")
