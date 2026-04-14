"""Across-sites evaluation: summarise one model's performance over all eval regions.

Loads pre-computed per-timestep metrics (metrics_region_time.nc and the
corresponding baseline file) from each eval region and produces summary plots
that place all sites on a shared x-axis so spatial variability in model skill
is immediately visible.

Plots generated
---------------
cross_site_rmse_boxplot
    Paired boxplots: one group per eval region, one box per source
    (model forecast + selected baseline).  Shows both the median error and
    the temporal spread within each site.

Usage
-----
uv run scripts/evaluation/run_across_sites.py \\
    --experiment mlp_rollout_L_KG_v1 \\
    --version version_1 \\
    --baseline era5 \\
    --lead_time 3h

# Multiple baselines (one pair per baseline per region)
uv run scripts/evaluation/run_across_sites.py \\
    --experiment mlp_rollout_L_KG_v1 \\
    --version version_1 \\
    --baseline era5 clim_hourly clim_smooth \\
    --lead_time 3h

Available --baseline names match the BASELINE_REGISTRY in run_evaluation.py:
    era5          → metrics_region_time_ERA5.nc
    clim_hourly   → metrics_region_time_hourly_clim.nc
    clim_smooth   → metrics_region_time_smoothed_clim.nc
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation.plots._colors import CASE_COLORS, DEFAULT_COLOR
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# Display label for each baseline registry key — must match CASE_COLORS keys.
_BASELINE_DISPLAY: dict[str, str] = {
    "era5": "ERA5",
    "clim_hourly": "hourly clim",
    "clim_smooth": "smoothed clim",
}

_ALL_EVALS = [
    "dahra",
    "antananarivo",
    "gobabeb",
    "johannesburg",
    "east_africa",
    "sahel_niger",
    "w_africa",
]

_PLOT_RC = {
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_case_label(experiment: str) -> str:
    if "rollout" in experiment.lower():
        return "MLP-Rollout"
    if "diag" in experiment.lower():
        return "MLP-Diag"
    return experiment


def _load_region_time(results_dir: Path, suffix: str = "") -> np.ndarray | None:
    """Load per-timestep RMSE array from metrics_region_time{suffix}.nc."""
    fname = f"metrics_region_time{suffix}.nc"
    path = results_dir / fname
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    if "rmse" not in ds:
        return None
    vals = ds["rmse"].values
    return vals[np.isfinite(vals)]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def cross_site_rmse_boxplot(
    model_vals_by_region: dict[str, np.ndarray],
    baselines_vals_by_region: dict[str, dict[str, np.ndarray]],
    model_label: str,
    metric: str = "rmse",
    save_path: Path = None,
    title: str = None,
) -> plt.Figure:
    """Paired boxplots: one group per eval region, boxes for model + baselines.

    Parameters
    ----------
    model_vals_by_region : dict[str, np.ndarray]
        ``{eval_region: rmse_values}`` — per-timestep RMSE for the model.
    baselines_vals_by_region : dict[str, dict[str, np.ndarray]]
        ``{baseline_label: {eval_region: rmse_values}}``.
    model_label : str
        Display label for the model (colour from CASE_COLORS).
    """
    plt.rcParams.update(_PLOT_RC)

    regions = list(model_vals_by_region.keys())
    baseline_labels = list(baselines_vals_by_region.keys())
    n_boxes = 1 + len(baseline_labels)  # model + each baseline
    box_width = 0.18
    group_spacing = n_boxes * box_width + 0.5  # tight packing + small gap
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

        # Model box
        model_vals = model_vals_by_region.get(region)
        if model_vals is not None and len(model_vals) > 0:
            x_pos = x_center + offsets[0]
            color = CASE_COLORS.get(model_label, DEFAULT_COLOR)
            bp = ax.boxplot(
                model_vals,
                positions=[x_pos],
                boxprops=dict(facecolor=color, alpha=0.65),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker=".", color=color, alpha=0.3, markersize=3),
                **_bp_kw,
            )
            if g_idx == 0:
                bp["boxes"][0].set_label(model_label)
                legend_handles.append(bp["boxes"][0])

        # Baseline boxes
        for b_idx, bl_label in enumerate(baseline_labels):
            bl_vals = baselines_vals_by_region[bl_label].get(region)
            if bl_vals is None or len(bl_vals) == 0:
                continue
            x_pos = x_center + offsets[1 + b_idx]
            color = CASE_COLORS.get(bl_label, DEFAULT_COLOR)
            bp = ax.boxplot(
                bl_vals,
                positions=[x_pos],
                boxprops=dict(facecolor=color, alpha=0.65),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker=".", color=color, alpha=0.3, markersize=3),
                **_bp_kw,
            )
            if g_idx == 0:
                bp["boxes"][0].set_label(bl_label)
                legend_handles.append(bp["boxes"][0])

    ax.set_xticks([g * group_spacing for g in range(len(regions))])
    ax.set_xticklabels(regions, rotation=20, ha="right")
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
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True, help="Experiment name, e.g. mlp_rollout_L_KG_v1")
    p.add_argument("--version", required=True, help="Version folder, e.g. version_1")
    p.add_argument(
        "--baseline",
        nargs="+",
        default=["era5"],
        help=f"Baseline(s) to compare against. Available: {list(_BASELINE_DISPLAY)}. (default: era5)",
    )
    p.add_argument("--lead_time", default="3h", help="Lead time tag (used in output filename, default: 3h)")
    p.add_argument(
        "--evals",
        nargs="+",
        default=_ALL_EVALS,
        help=f"Eval regions to include (default: all {len(_ALL_EVALS)})",
    )
    p.add_argument("--metric", default="rmse", help="Metric to plot (default: rmse)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run_across_sites():
    args = parse_args()
    case_label = _detect_case_label(args.experiment)

    out_dir = PATHS.results_dir(args.experiment, args.version, "across_sites")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Case       : {case_label}")
    print(f"Baselines  : {args.baseline}")
    print(f"Eval regions: {args.evals}")

    # -------------------------------------------------------------------------
    # Load per-region metrics
    # -------------------------------------------------------------------------
    model_vals: dict[str, np.ndarray] = {}
    baselines_vals: dict[str, dict[str, np.ndarray]] = {
        _BASELINE_DISPLAY[b]: {} for b in args.baseline if b in _BASELINE_DISPLAY
    }

    for eval_region in args.evals:
        results_dir = PATHS.results_dir(args.experiment, args.version, eval_region)

        vals = _load_region_time(results_dir, suffix="")
        if vals is None:
            print(f"  [skip] {eval_region}: metrics_region_time.nc not found — run run_evaluation.py first")
            continue
        model_vals[eval_region] = vals
        print(f"  [ok]   {eval_region}: {len(vals)} model timesteps")

        for b_key in args.baseline:
            bl_display = _BASELINE_DISPLAY.get(b_key)
            if bl_display is None:
                print(f"  [warn] unknown baseline key {b_key!r}")
                continue
            bl_file = bl_display.replace(" ", "_")
            bl_vals = _load_region_time(results_dir, suffix=f"_{bl_file}")
            if bl_vals is None:
                print(f"  [skip] {eval_region} / {bl_display}: metrics_region_time_{bl_file}.nc not found")
                continue
            baselines_vals[bl_display][eval_region] = bl_vals

    if not model_vals:
        print("No model metrics found. Exiting.")
        return

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    bl_str = "_".join(_BASELINE_DISPLAY.get(b, b) for b in args.baseline)
    save_path = out_dir / f"cross_site_{args.metric}_lt{args.lead_time}_vs_{bl_str}.png"

    if save_path.exists() and not args.overwrite:
        print(f"Plot exists (--overwrite to regenerate): {save_path}")
    else:
        cross_site_rmse_boxplot(
            model_vals_by_region=model_vals,
            baselines_vals_by_region=baselines_vals,
            model_label=case_label,
            metric=args.metric,
            save_path=save_path,
            title=f"{case_label}  —  {args.metric.upper()} across sites  (lt={args.lead_time})",
        )

    print("\nDone.")


if __name__ == "__main__":
    run_across_sites()
