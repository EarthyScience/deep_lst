"""Cross-site summary plots: RMSE, ACC, and ACC+NSE distributions across eval regions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deep_lst.evaluation.plots._design import CASE_COLORS, DEFAULT_COLOR, RC


def _draw(ax, vals, x_pos, color, width, violin, bp_kw):
    """Draw one box or violin; return an artist usable as a legend handle."""
    if violin:
        vp = ax.violinplot(vals, positions=[x_pos], widths=width * 1.2, showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.65)
        for part in ("cmins", "cmaxes", "cbars"):
            vp[part].set_color(color)
            vp[part].set_linewidth(1.0)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.5)
        return vp["bodies"][0]
    else:
        bp = ax.boxplot(
            vals,
            positions=[x_pos],
            boxprops=dict(facecolor=color, alpha=0.65),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker=".", color=color, alpha=0.3, markersize=3),
            **bp_kw,
        )
        return bp["boxes"][0]


def cross_site_rmse_boxplot(
    model_vals_by_region: dict[str, np.ndarray],
    baselines_vals_by_region: dict[str, dict[str, np.ndarray]],
    model_label: str,
    metric: str = "rmse",
    save_path: Path = None,
    title: str = None,
    log_y: bool = True,
    violin: bool = False,
) -> plt.Figure:
    """Paired boxplots: one group per eval region, boxes for model + baselines.

    Parameters
    ----------
    model_vals_by_region : dict[str, np.ndarray]
        ``{region_label: rmse_values}`` — per-timestep RMSE for the model.
    baselines_vals_by_region : dict[str, dict[str, np.ndarray]]
        ``{baseline_label: {region_label: rmse_values}}``.
    model_label : str
        Display label for the model (colour from CASE_COLORS).
    log_y : bool
        Log-scale y-axis (default: True).
    """
    plt.rcParams.update(RC)

    regions = list(model_vals_by_region.keys())
    baseline_labels = list(baselines_vals_by_region.keys())
    n_boxes = 1 + len(baseline_labels)
    box_width = 0.18
    group_spacing = n_boxes * box_width + 0.5
    offsets = np.linspace(-(n_boxes - 1) / 2, (n_boxes - 1) / 2, n_boxes) * (box_width + 0.04)

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * group_spacing * 1.5 - 4.5), 5))

    _bp_kw = dict(
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        manage_ticks=False,
    )

    legend_handles = []

    for g_idx, region in enumerate(regions):
        x_center = g_idx * group_spacing

        model_vals = model_vals_by_region.get(region)
        if model_vals is not None and len(model_vals) > 0:
            x_pos = x_center + offsets[0]
            color = CASE_COLORS.get(model_label, DEFAULT_COLOR)
            handle = _draw(ax, model_vals, x_pos, color, box_width, violin, _bp_kw)
            if g_idx == 0:
                handle.set_label(model_label)
                legend_handles.append(handle)

        for b_idx, bl_label in enumerate(baseline_labels):
            bl_vals = baselines_vals_by_region[bl_label].get(region)
            if bl_vals is None or len(bl_vals) == 0:
                continue
            x_pos = x_center + offsets[1 + b_idx]
            color = CASE_COLORS.get(bl_label, DEFAULT_COLOR)
            handle = _draw(ax, bl_vals, x_pos, color, box_width, violin, _bp_kw)
            if g_idx == 0:
                handle.set_label(bl_label)
                legend_handles.append(handle)

    ax.set_xticks([g * group_spacing for g in range(len(regions))])
    ax.set_xticklabels(regions, rotation=20, ha="right")
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(f"{metric.upper()} (K)")
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    bl_str = " + ".join(baseline_labels) if baseline_labels else "no baseline"
    ax.set_title(title or f"{model_label}  —  {metric.upper()} across sites  (vs {bl_str})")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
        print(f"Saved: {save_path}")

    return fig


def cross_site_acc_boxplot(
    acc_vals_by_region: dict[str, np.ndarray],
    save_path: Path = None,
    violin: bool = False,
) -> plt.Figure:
    """One boxplot per eval region showing the spatial distribution of ACC.

    Each box shows the distribution of per-pixel ACC values across the region.
    ACC is the anomaly correlation coefficient: Pearson r between (pred - clim)
    and (obs - clim), computed per pixel over the evaluation period.

    Parameters
    ----------
    acc_vals_by_region : dict[str, np.ndarray]
        ``{region_label: acc_per_pixel}`` — flattened, NaN-free ACC values.
    model_label : str
        Display label for the model (colour from CASE_COLORS).
    """
    plt.rcParams.update(RC)

    regions = list(acc_vals_by_region.keys())
    box_width = 0.3
    group_spacing = box_width + 0.4

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * group_spacing * 1.2), 4.5))

    color = DEFAULT_COLOR
    _bp_kw = dict(
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        manage_ticks=False,
    )

    for g_idx, region in enumerate(regions):
        vals = acc_vals_by_region.get(region)
        if vals is None or len(vals) == 0:
            continue
        x_pos = g_idx * group_spacing
        _draw(ax, vals, x_pos, color, box_width, violin, _bp_kw)

    ax.set_xticks([g * group_spacing for g in range(len(regions))])
    ax.set_xticklabels(regions, rotation=20, ha="right", fontweight="bold")
    ax.set_ylabel("Anomaly Correlation Coefficient", fontweight="bold")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=0, label="ACC=0.5")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"Saved: {save_path}")

    return fig


def cross_site_skill_boxplot(
    acc_vals_by_region: dict[str, np.ndarray],
    nse_vals_by_region: dict[str, np.ndarray],
    save_path: Path = None,
    violin: bool = False,
) -> plt.Figure:
    """Side-by-side ACC and NSE violin/box plots per eval region.

    Both metrics are clipped at -0.1 (lower bound). Reference lines at ACC=0.5
    and NSE=0 are drawn.

    Parameters
    ----------
    acc_vals_by_region : dict[str, np.ndarray]
        ``{region_label: acc_per_pixel}`` — per-pixel ACC, NaN-free.
    nse_vals_by_region : dict[str, np.ndarray]
        ``{region_label: nse_per_pixel}`` — per-pixel NSE, NaN-free.
    """
    plt.rcParams.update(RC)

    regions = list(acc_vals_by_region.keys())
    box_width = 0.22
    group_spacing = 2 * box_width + 0.25
    offsets = np.array([-1, 1]) * (box_width / 2 + 0.02)

    acc_color = DEFAULT_COLOR
    nse_color = "orchid"

    _bp_kw = dict(
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        manage_ticks=False,
    )

    fig, ax = plt.subplots(figsize=(max(5, len(regions) * group_spacing * 1.3), 4.5))

    half = group_spacing / 2
    stripe_colors = ["#f5f5f5", "#ebebeb"]
    for g_idx in range(len(regions)):
        x_center = g_idx * group_spacing
        ax.axvspan(x_center - half, x_center + half, color=stripe_colors[g_idx % 2], zorder=0)

    legend_handles = []
    for g_idx, region in enumerate(regions):
        x_center = g_idx * group_spacing

        acc_vals = acc_vals_by_region.get(region)
        if acc_vals is not None and len(acc_vals) > 0:
            clipped = np.clip(acc_vals, -0.1, None)
            h = _draw(ax, clipped, x_center + offsets[0], acc_color, box_width, violin, _bp_kw)
            if g_idx == 0:
                h.set_label("ACC")
                legend_handles.append(h)

        nse_vals = nse_vals_by_region.get(region)
        if nse_vals is not None and len(nse_vals) > 0:
            clipped = np.clip(nse_vals, -0.1, None)
            h = _draw(ax, clipped, x_center + offsets[1], nse_color, box_width, violin, _bp_kw)
            if g_idx == 0:
                h.set_label("NSE")
                legend_handles.append(h)

    ax.axhline(0.5, color=acc_color, linewidth=0.9, linestyle="--", zorder=0)
    ax.axhline(0.0, color=nse_color, linewidth=0.9, linestyle="--", zorder=0)

    ax.set_xticks([g * group_spacing for g in range(len(regions))])
    ax.set_xticklabels(regions, rotation=20, ha="center", fontweight="bold")
    ax.set_ylabel("Skill score", fontweight="bold")
    ax.set_ylim(bottom=-0.1)
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=False)
        print(f"Saved: {save_path}")

    return fig
