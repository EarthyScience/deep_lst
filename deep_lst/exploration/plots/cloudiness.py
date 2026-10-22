"""Cloudiness / NaN-fraction plots for exploration.

plot_cloudiness_polar(nan_frac, period_labels, save_path)
    2×4 polar bar panels, one per eval site. Bar angle = test period,
    bar length = mean NaN fraction. 100% periods red, lowest per site purple.

plot_cloudiness_heatmap(nan_frac, period_labels, save_path)
    Sites × periods heatmap. Color = mean NaN fraction; 100% cells marked with ×,
    lowest NaN fraction per site highlighted in purple.
"""

import calendar
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from deep_lst.exploration.plots._design import RC


def _month_period_ticks(period_names, period_labels):
    """First period index and month abbreviation for each calendar month."""
    seen = {}
    for i, name in enumerate(period_names):
        month = int(period_labels[name].split("–")[0].strip().split("-")[1])
        if month not in seen:
            seen[month] = i
    return list(seen.values()), [calendar.month_abbr[m] for m in seen]


def plot_cloudiness_polar(nan_frac, period_labels, save_path):
    plt.rcParams.update(RC)

    eval_sites = list(nan_frac.keys())
    period_names = list(next(iter(nan_frac.values())).keys())
    n = len(period_names)
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.85

    tick_pos, tick_labels = _month_period_ticks(period_names, period_labels)
    tick_angles_deg = [360 * i / n for i in tick_pos]

    fig, axes = plt.subplots(2, 4, subplot_kw={"projection": "polar"}, figsize=(18, 12))
    for ax, site in zip(axes.flat, eval_sites):
        means = np.array([nan_frac[site][name].mean() for name in period_names])
        min_idx = int(np.nanargmin(means))

        colors = np.where(means >= 1.0, "#e74c3c", "#4a90d9").tolist()
        colors[min_idx] = "#8e44ad"

        ax.bar(thetas, means, width=width, color=colors, alpha=0.85)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(["50%", "100%"], fontsize=11)
        ax.set_thetagrids(tick_angles_deg, tick_labels, fontsize=12)
        ax.set_title(site.title(), fontsize=14, pad=16)

    plt.tight_layout(h_pad=5, w_pad=4)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_cloudiness_heatmap(nan_frac, period_labels, save_path):
    plt.rcParams.update(RC)

    eval_sites = list(nan_frac.keys())
    period_names = list(next(iter(nan_frac.values())).keys())

    data = np.array([[nan_frac[site][name].mean() for name in period_names] for site in eval_sites])

    tick_pos, tick_labels = _month_period_ticks(period_names, period_labels)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    # 100% cells: red ×
    rows, cols = np.where(data >= 1.0)
    for r, c in zip(rows, cols):
        ax.text(c, r, "×", ha="center", va="center", color="#e74c3c", fontsize=11, fontweight="bold")

    # Lowest NaN fraction per site (row): purple tile
    for r in range(len(eval_sites)):
        min_col = int(np.nanargmin(data[r]))
        ax.add_patch(
            mpatches.Rectangle(
                (min_col - 0.5, r - 0.5),
                1,
                1,
                linewidth=0,
                facecolor="#8e44ad",
                zorder=3,
                alpha=0.75,
            )
        )

    ax.set_yticks(range(len(eval_sites)))
    ax.set_yticklabels(eval_sites)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Test period")
    plt.colorbar(
        im, ax=ax, label="Mean NaN fraction", fraction=0.02, pad=0.02, format=plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig
