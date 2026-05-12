"""Split-diagonal bubble matrix for sweep experiment overview.

sweep_overview(grid_data, metric_top, metric_bot, eval_name, out_path)
    4×4 matrix: rows = model×size, cols = stratification×loss.
    Each cell has a white background split by a dark-gray diagonal.
    Bubbles in each half are colored by metric value (gradient colormap,
    consistent across all cells per metric) and sized by 2×temporal std.
    Values are annotated inside bubbles; no colorbars needed.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROWS = [
    ("diag", "L"),
    ("diag", "XL"),
    ("rollout", "L"),
    ("rollout", "XL"),
]

COLS = [
    ("KG", "crps"),
    ("KG", "mse"),
    ("KGLC", "crps"),
    ("KGLC", "mse"),
]

# Bubble fill colormaps: single-hue gradients per metric
TOP_CMAP = plt.get_cmap("Reds")
BOT_CMAP = plt.get_cmap("Blues")

R_MAX = 0.28  # max bubble radius in data units


def _triangle_verts(x0, y0, corner):
    if corner == "upper":
        return np.array([[x0, y0 + 1], [x0 + 1, y0], [x0 + 1, y0 + 1]])
    return np.array([[x0, y0], [x0 + 1, y0], [x0, y0 + 1]])


def _centroid(x0, y0, corner):
    return _triangle_verts(x0, y0, corner).mean(axis=0)


def sweep_overview(
    grid_data: dict,
    metric_top: str,
    metric_bot: str,
    eval_name: str,
    out_path: Path,
) -> None:
    """Draw and save the sweep overview plot.

    Parameters
    ----------
    grid_data : dict[(row_idx, col_idx), (top_mean, top_std, bot_mean, bot_std)]
        Keyed by (row, col) indices into ROWS / COLS.
        Any nan value is treated as missing.
    metric_top, metric_bot : str
        Metric names for upper-right / lower-left triangles.
    eval_name : str
        Used in the figure title.
    out_path : Path
        Output file path (PDF or PNG).
    """
    nrows = len(ROWS)
    ncols = len(COLS)

    fin = np.isfinite

    # Separate color norms per model family so diag and rollout each use
    # the full colormap range independently.
    diag_rows = {ri for ri, (fam, _) in enumerate(ROWS) if fam == "diag"}
    rollout_rows = {ri for ri, (fam, _) in enumerate(ROWS) if fam == "rollout"}

    def _norm(keys):
        vals_top = [grid_data[k][0] for k in keys if k in grid_data and fin(grid_data[k][0])]
        vals_bot = [grid_data[k][2] for k in keys if k in grid_data and fin(grid_data[k][2])]
        t = plt.Normalize(min(vals_top), max(vals_top)) if vals_top else plt.Normalize(0, 1)
        b = plt.Normalize(min(vals_bot), max(vals_bot)) if vals_bot else plt.Normalize(0, 1)
        return t, b

    diag_keys = [(ri, ci) for ri in diag_rows for ci in range(len(COLS))]
    rollout_keys = [(ri, ci) for ri in rollout_rows for ci in range(len(COLS))]
    diag_top_norm, diag_bot_norm = _norm(diag_keys)
    rollout_top_norm, rollout_bot_norm = _norm(rollout_keys)

    all_top_std = [v[1] for v in grid_data.values() if fin(v[1])]
    all_bot_std = [v[3] for v in grid_data.values() if fin(v[3])]
    top_std_max = max(all_top_std) if all_top_std else 1.0
    bot_std_max = max(all_bot_std) if all_bot_std else 1.0

    fig, ax = plt.subplots(figsize=(ncols * 1.4 + 1.8, nrows * 1.4 + 1.8))
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, ncols + 0.05)
    ax.set_ylim(-0.05, nrows + 0.05)
    ax.axis("off")

    for ri in range(nrows):
        for ci in range(ncols):
            x0, y0 = ci, nrows - 1 - ri
            top_mean, top_std, bot_mean, bot_std = grid_data.get((ri, ci), (float("nan"),) * 4)

            # White triangle backgrounds with dark-gray diagonal separator
            ax.add_patch(
                mpatches.Polygon(
                    _triangle_verts(x0, y0, "upper"), closed=True, facecolor="white", edgecolor="#555", lw=0.6
                )
            )
            ax.add_patch(
                mpatches.Polygon(
                    _triangle_verts(x0, y0, "lower"), closed=True, facecolor="white", edgecolor="#555", lw=0.6
                )
            )

            # Black tile border on top
            ax.add_patch(mpatches.Rectangle((x0, y0), 1, 1, fill=False, edgecolor="black", lw=1.2))

            top_norm = diag_top_norm if ri in diag_rows else rollout_top_norm
            bot_norm = diag_bot_norm if ri in diag_rows else rollout_bot_norm

            # Upper-right bubble: metric_top (RMSE)
            if fin(top_mean) and fin(top_std):
                cx, cy = _centroid(x0, y0, "upper")
                r = max(R_MAX * (2 * top_std) / (2 * top_std_max), 0.04)
                fc = TOP_CMAP(top_norm(top_mean))
                ax.add_patch(mpatches.Circle((cx, cy), radius=r, facecolor=fc, edgecolor="black", lw=0.5))
                ax.text(
                    cx, cy, f"{top_mean:.2f}", ha="center", va="center", fontsize=6.5, fontweight="bold", color="black"
                )

            # Lower-left bubble: metric_bot (MAE)
            if fin(bot_mean) and fin(bot_std):
                cx, cy = _centroid(x0, y0, "lower")
                r = max(R_MAX * (2 * bot_std) / (2 * bot_std_max), 0.04)
                fc = BOT_CMAP(bot_norm(bot_mean))
                ax.add_patch(mpatches.Circle((cx, cy), radius=r, facecolor=fc, edgecolor="black", lw=0.5))
                ax.text(
                    cx, cy, f"{bot_mean:.2f}", ha="center", va="center", fontsize=6.5, fontweight="bold", color="black"
                )

    # Row labels
    prev_fam = None
    for ri, (fam, sz) in enumerate(ROWS):
        y0 = nrows - 1 - ri
        ax.text(-0.10, y0 + 0.5, sz, ha="right", va="center", fontsize=9)
        if fam != prev_fam:
            ax.text(-0.60, y0 + 0.0, fam, ha="right", va="center", fontsize=9, fontweight="bold", rotation=90)
            prev_fam = fam

    # Col labels
    prev_st = None
    for ci, (st, ls) in enumerate(COLS):
        x0 = ci
        ax.text(x0 + 0.5, nrows + 0.06, ls, ha="center", va="bottom", fontsize=9)
        if st != prev_st:
            ax.text(x0 + 1.0, nrows + 0.38, st, ha="center", va="bottom", fontsize=9, fontweight="bold")
            prev_st = st

    # Metric corner labels (one cell, bottom-left corner of whole grid)
    ax.text(
        ncols - 0.05, nrows - 0.05, metric_top.upper(), ha="right", va="top", fontsize=7, color="#333", style="italic"
    )
    ax.text(0.05, 0.05, metric_bot.upper(), ha="left", va="bottom", fontsize=7, color="#333", style="italic")

    # Bubble-size legend (below the grid)
    legend_items = [(1.0, "max 2σ"), (0.5, "½ max")]
    lx_start = 0.5
    ly = -0.55
    ax.text(lx_start - 0.15, ly, "bubble size = 2×std:", ha="left", va="center", fontsize=7, color="#333")
    lx = lx_start + 1.5
    for scale, label in legend_items:
        r = R_MAX * scale
        ax.add_patch(mpatches.Circle((lx, ly), radius=r, facecolor="lightgray", edgecolor="black", lw=0.6))
        ax.text(lx + r + 0.08, ly, label, ha="left", va="center", fontsize=7, color="#333")
        lx += 1.4

    fig.suptitle(
        f"Sweep overview — {eval_name}\n"
        f"▲ {metric_top.upper()} (warm)   ▽ {metric_bot.upper()} (cool)   color scaled per model family",
        fontsize=10,
        y=1.02,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)
