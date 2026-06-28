"""Generic two-panel (Day | Night) months x columns heatmap.

Shared renderer for the forecast-limit and monthly-score heatmaps: rows = months (ascending),
columns = regions or classes, one panel per day/night. The caller supplies the value column, the
colormap/norm and colorbar, plus optional latitude top-axis labels and a train/eval separator.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from deep_lst.evaluation.plots._design import RC

MONTHS = list(range(1, 13))
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _latitude_labels(df, col_order, training_label):
    """Top-axis tick labels: centre latitude per column, blank for the training column."""
    latitude = df.groupby("region")["latitude"].first()
    return ["" if name == training_label else f"{latitude[name]:.1f}°" for name in col_order]


def continuous_norm(vmin, vmax, cmap_name="viridis"):
    """Continuous sequential colormap (grey for missing) + linear norm for monthly-score heatmaps."""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")
    return cmap, Normalize(vmin=vmin, vmax=vmax)


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
