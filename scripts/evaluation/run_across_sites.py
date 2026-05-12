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

import numpy as np
import xarray as xr

from deep_lst.evaluation.plots.across_sites import cross_site_acc_boxplot, cross_site_rmse_boxplot
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# Display label for each baseline registry key — must match CASE_COLORS keys.
_BASELINE_DISPLAY: dict[str, str] = {
    "era5": "ERA5",
    "clim_hourly": "hourly clim",
    "clim_smooth": "smoothed clim",
    "persistence": "persistence",
}

_ALL_EVALS = [
    "dahra",
    "antananarivo",
    "bouamir",
    "gobabeb",
    "johannesburg",
    "east_africa",
    "sahel_niger",
    "w_africa",
    "yangambi",
]

_REGION_DISPLAY: dict[str, str] = {
    "dahra": "Dahra",
    "antananarivo": "Antananarivo",
    "bouamir": "Bouamir",
    "gobabeb": "Gobabeb",
    "johannesburg": "Johannesburg",
    "east_africa": "Passage",
    "sahel_niger": "Sahel",
    "w_africa": "West Africa",
    "yangambi": "Yangambi",
}

# Maps file-based display label (from _BASELINE_DISPLAY) → plot legend label.
_BASELINE_PLOT_LABEL: dict[str, str] = {
    "ERA5": "ERA5",
    "hourly clim": "Climatology",
    "smoothed clim": "Climatology (smooth)",
    "persistence": "Persistence",
}

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
        return "Deep LST"
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


def _load_region_space_acc(results_dir: Path) -> np.ndarray | None:
    """Load flattened per-pixel ACC array from metrics_region_space_acc.nc."""
    path = results_dir / "metrics_region_space_acc.nc"
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    if "correlation" not in ds:
        return None
    vals = ds["correlation"].values.ravel()
    return vals[np.isfinite(vals)]


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
    p.add_argument("--violin", action="store_true", help="Use violin plots instead of boxplots")
    return p.parse_args()


def run_across_sites():
    args = parse_args()
    case_label = "Deep LST"

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

    acc_vals: dict[str, np.ndarray] = {}

    for eval_region in args.evals:
        results_dir = PATHS.results_dir(args.experiment, args.version, eval_region)

        vals = _load_region_time(results_dir, suffix="")
        if vals is None:
            print(f"  [skip] {eval_region}: metrics_region_time.nc not found — run run_evaluation.py first")
            continue
        model_vals[eval_region] = vals
        print(f"  [ok]   {eval_region}: {len(vals)} model timesteps")

        acc = _load_region_space_acc(results_dir)
        if acc is not None:
            acc_vals[eval_region] = acc
            print(f"  [ok]   {eval_region}: {len(acc)} ACC pixels")

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

    # Apply display name mappings before plotting
    model_vals = {_REGION_DISPLAY.get(r, r): v for r, v in model_vals.items()}
    acc_vals = {_REGION_DISPLAY.get(r, r): v for r, v in acc_vals.items()}
    baselines_vals = {
        _BASELINE_PLOT_LABEL.get(bl, bl): {_REGION_DISPLAY.get(r, r): v for r, v in region_vals.items()}
        for bl, region_vals in baselines_vals.items()
    }

    # -------------------------------------------------------------------------
    # RMSE boxplot
    # -------------------------------------------------------------------------
    bl_str = "_".join(_BASELINE_DISPLAY.get(b, b) for b in args.baseline)
    rmse_path = out_dir / f"cross_site_{args.metric}_lt{args.lead_time}_vs_{bl_str}.pdf"

    if rmse_path.exists() and not args.overwrite:
        print(f"Plot exists (--overwrite to regenerate): {rmse_path}")
    else:
        cross_site_rmse_boxplot(
            model_vals_by_region=model_vals,
            baselines_vals_by_region=baselines_vals,
            model_label=case_label,
            metric=args.metric,
            save_path=rmse_path,
            title=f"{case_label}  —  {args.metric.upper()} across sites  (lt={args.lead_time})",
            violin=args.violin,
        )

    # -------------------------------------------------------------------------
    # ACC boxplot (requires metrics_region_space_acc.nc — run_evaluation.py with
    # --baselines clim_hourly must have been run first)
    # -------------------------------------------------------------------------
    if acc_vals:
        acc_path = out_dir / f"cross_site_acc_lt{args.lead_time}.pdf"
        if acc_path.exists() and not args.overwrite:
            print(f"Plot exists (--overwrite to regenerate): {acc_path}")
        else:
            cross_site_acc_boxplot(
                acc_vals_by_region=acc_vals,
                save_path=acc_path,
                violin=args.violin,
            )
    else:
        print(
            "  [skip] ACC plot: no metrics_region_space_acc.nc found — run run_evaluation.py with --baselines clim_hourly"
        )

    print("\nDone.")


if __name__ == "__main__":
    run_across_sites()
