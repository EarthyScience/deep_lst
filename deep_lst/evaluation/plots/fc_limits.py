"""Two-panel (Day | Night) forecast-limit-by-month heatmap.

Rows = months (ascending), columns = regions (training holdout at x=0, then eval
regions sorted by descending latitude). Bottom tick = region name, top tick = the
region's centre latitude. Cell colour = forecast limit in days on a discretised
sequential scale (light = longer limit); missing cells are drawn grey.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from deep_lst.evaluation.plots._design import RC
from deep_lst.evaluation.plots.month_grid import (
    MONTH_LABELS,
    MONTHS,
    _cell_matrix,
    _latitude_labels,
    month_grid_heatmap,
)

# Sequential map sampled between these fractions so the darkest cell is not pure black
# and the lightest not pure white (which would blend into the page).
CMAP_NAME = "plasma"
CMAP_RANGE = (0.0, 0.85)


def _limit_colormap(horizon_days):
    """Discretised plasma colormap + integer-day BoundaryNorm shared by both panels/plots."""
    n_levels = int(round(horizon_days))
    capped_colors = plt.get_cmap(CMAP_NAME)(np.linspace(CMAP_RANGE[0], CMAP_RANGE[1], n_levels))
    cmap = ListedColormap(capped_colors)
    cmap.set_bad("lightgrey")
    boundaries = np.arange(0.5, n_levels + 1.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, n_levels


def _limit_colorbar(fig, mappable, axes, n_levels):
    """Forecast-limit colorbar with <1 / >10 endpoint labels, shared by both plots."""
    colorbar = fig.colorbar(mappable, ax=axes, ticks=range(1, n_levels + 1), fraction=0.046, pad=0.04)
    colorbar.set_label("Forecast limit [days]", fontweight="bold")
    labels = [str(v) for v in range(1, n_levels + 1)]
    labels[0], labels[-1] = "<1", ">10"
    colorbar.set_ticklabels(labels)


def _limit_cbar(horizon_days):
    """(cmap, norm, ticks, ticklabels) for the discretised forecast-limit colorbar (<1 .. >10 days)."""
    cmap, norm, n_levels = _limit_colormap(horizon_days)
    ticks = range(1, n_levels + 1)
    ticklabels = [str(v) for v in ticks]
    ticklabels[0], ticklabels[-1] = "<1", ">10"
    return cmap, norm, ticks, ticklabels


def forecast_limit_month_heatmap(df, region_order, training_label, horizon_days, save_path):
    """df: tidy frame with columns region, month, daynight, limit_days, latitude."""
    cmap, norm, ticks, ticklabels = _limit_cbar(horizon_days)
    month_grid_heatmap(
        df,
        region_order,
        "limit_days",
        save_path,
        cmap=cmap,
        norm=norm,
        cbar_label="Forecast limit [days]",
        cbar_ticks=ticks,
        cbar_ticklabels=ticklabels,
        top_labels=_latitude_labels(df, region_order, training_label),
        separator_x=0.5,
        col_gridlines=True,
        figsize_per_col=1.2,
        figsize_base=2.0,
    )


# Validity (clear-sky %) -> scatter marker area (points^2).
SIZE_MIN = 15.0
SIZE_MAX = 360.0
SIZE_LEGEND_PCT = (25, 50, 75, 100)


def _validity_size(validity):
    """Map clear-sky validity in [0, 100] to scatter marker area; NaN -> SIZE_MIN."""
    area = SIZE_MIN + np.clip(validity, 0, 100) / 100.0 * (SIZE_MAX - SIZE_MIN)
    return np.where(np.isfinite(area), area, SIZE_MIN)


def forecast_limit_month_bubbles(df, region_order, training_label, horizon_days, save_path):
    """Two-panel (Day | Night) bubble matrix: colour = forecast limit, size = clear-sky validity.

    Same months x regions layout as the heatmap, but each cell is a circle whose area encodes
    observation validity, so low-validity (less trustworthy) limits read as small markers.
    Missing cells (no forecast limit) are drawn as a small grey x.
    """
    plt.rcParams.update(RC)

    cmap, norm, n_levels = _limit_colormap(horizon_days)
    latitude_labels = _latitude_labels(df, region_order, training_label)
    rows, cols = np.mgrid[0 : len(MONTHS), 0 : len(region_order)]

    fig, axes = plt.subplots(1, 2, figsize=(1.2 * len(region_order) + 2, 6), sharey=True)
    points = None
    for ax, kind in zip(axes, ("day", "night")):
        limit = _cell_matrix(df, region_order, kind, "limit_days").ravel()
        validity = _cell_matrix(df, region_order, kind, "validity").ravel()
        x, y = cols.ravel(), rows.ravel()

        present = np.isfinite(limit)
        ax.scatter(x[~present], y[~present], marker="x", c="lightgrey", s=20, linewidths=1.0)
        points = ax.scatter(
            x[present],
            y[present],
            c=limit[present],
            s=_validity_size(validity[present]),
            cmap=cmap,
            norm=norm,
            edgecolors="black",
            linewidths=0.4,
        )

        ax.set_xticks(range(len(region_order)))
        ax.set_xticklabels(region_order, rotation=45, ha="right", fontweight="bold")
        ax.set_title(kind.capitalize(), pad=36, fontweight="bold")
        ax.set_xlim(-0.6, len(region_order) - 0.4)
        ax.set_ylim(-0.6, len(MONTHS) - 0.4)
        ax.axvline(0.5, color="gray", linestyle="-", linewidth=1.5)  # training | eval separator
        ax.grid(color="gray", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

        top = ax.secondary_xaxis("top")
        top.set_xticks(range(len(region_order)))
        top.set_xticklabels(latitude_labels, rotation=45, ha="left")

    axes[0].set_yticks(range(len(MONTHS)))
    axes[0].set_yticklabels(MONTH_LABELS, fontweight="bold")

    _limit_colorbar(fig, points, axes, n_levels)

    size_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.6",
            markeredgecolor="black",
            markersize=float(np.sqrt(_validity_size(pct))),
            label=f"{pct}%",
        )
        for pct in SIZE_LEGEND_PCT
    ]
    fig.legend(
        handles=size_handles,
        title="Clear-sky validity",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=len(SIZE_LEGEND_PCT),
        frameon=False,
        handletextpad=0.2,
        columnspacing=1.2,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")


def forecast_limit_validity_scatter(df, save_path, color_col="region"):
    """Per-cell scatter of clear-sky validity (x) vs forecast limit (y), one panel per day/night.

    Each point is one (region, month) cell, coloured by `color_col`. Reached-limit cells are filled,
    right-censored cells (limit at the horizon) hollow. Each panel annotates Pearson r and Spearman
    rho over all finite cells (censored included).
    """
    plt.rcParams.update(RC)
    groups = sorted(df[color_col].dropna().unique())
    palette = plt.get_cmap("tab20" if len(groups) > 10 else "tab10")
    color_by = {group: palette(i % palette.N) for i, group in enumerate(groups)}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, kind in zip(axes, ("day", "night")):
        panel = df[df["daynight"] == kind]
        for group in groups:
            sub = panel[panel[color_col] == group]
            reached = sub[~sub["censored"].astype(bool)]
            censored = sub[sub["censored"].astype(bool)]
            ax.scatter(
                reached["validity"],
                reached["limit_days"],
                s=32,
                color=color_by[group],
                edgecolors="black",
                linewidths=0.3,
                label=group,
            )
            ax.scatter(
                censored["validity"],
                censored["limit_days"],
                s=32,
                facecolors="none",
                edgecolors=color_by[group],
                linewidths=1.0,
            )

        reached_cells = panel[~panel["censored"].astype(bool)]
        finite = reached_cells[
            ["validity", "limit_days"]
        ].dropna()  # censored excluded: their limit is only a lower bound
        if len(finite) >= 3:
            r = finite.corr(method="pearson").iloc[0, 1]
            rho = finite.corr(method="spearman").iloc[0, 1]
            ax.text(
                0.04,
                0.96,
                f"r = {r:.2f}\n" + r"$\rho$ = " + f"{rho:.2f}\nn = {len(finite)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
            )
        ax.set_title(kind.capitalize(), fontweight="bold")
        ax.set_xlabel("Clear-sky validity [%]", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Forecast limit [days]", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, title=color_col)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"[written] {save_path}")


def forecast_limit_class_heatmap(df, class_order, horizon_days, save_path):
    """Two-panel (Day | Night) forecast-limit heatmap over land-cover classes (x) x months (y).

    Same renderer as the region heatmap but columns are classes (no latitude axis, no train/eval
    separator). df: tidy frame with region (= class label), month, daynight, limit_days.
    """
    cmap, norm, ticks, ticklabels = _limit_cbar(horizon_days)
    month_grid_heatmap(
        df,
        class_order,
        "limit_days",
        save_path,
        cmap=cmap,
        norm=norm,
        cbar_label="Forecast limit [days]",
        cbar_ticks=ticks,
        cbar_ticklabels=ticklabels,
        figsize_per_col=0.55,
        figsize_base=3.0,
    )
