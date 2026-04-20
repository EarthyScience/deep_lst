"""Sweep overview plot: split-diagonal bubble matrix across all 16 experiments.

Loads metrics_total.nc (mean) and metrics_region_time.nc (temporal std) for
each experiment, then calls sweep_overview() to produce the matrix plot.

Usage
-----
uv run scripts/evaluation/run_sweep_overview.py
    --eval train_KG_S --version version_0

uv run scripts/evaluation/run_sweep_overview.py \\
    --eval train_KG_S --version version_0 \\
    --metric_top rmse --metric_bot mae \\
    --out results/sweep_overview.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from deep_lst.evaluation.plots.sweep import COLS, ROWS, sweep_overview
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def load_metric(experiment: str, version: str, eval_name: str, metric: str):
    """Return (mean, std) for `metric`, or (nan, nan) on missing data."""
    results_dir = PATHS.results_dir(experiment, version, eval_name)
    total_path = results_dir / "metrics_total.nc"
    rt_path = results_dir / "metrics_region_time.nc"

    if not total_path.exists():
        return float("nan"), float("nan")

    try:
        total = xr.open_dataset(total_path)
        if metric not in total:
            return float("nan"), float("nan")
        mean_val = float(total[metric].values.flat[0])
    except Exception:
        return float("nan"), float("nan")

    if not rt_path.exists():
        return mean_val, float("nan")

    try:
        rt = xr.open_dataset(rt_path)
        if metric not in rt:
            return mean_val, float("nan")
        std_val = float(rt[metric].std("valid_time").values.flat[0])
    except Exception:
        return mean_val, float("nan")

    return mean_val, std_val


def build_grid(version: str, eval_name: str, metric_top: str, metric_bot: str) -> dict:
    grid = {}
    for ri, (fam, sz) in enumerate(ROWS):
        for ci, (st, ls) in enumerate(COLS):
            exp = f"mlp_{fam}_{sz}_{st}_{ls}_h3_v1"
            top_mean, top_std = load_metric(exp, version, eval_name, metric_top)
            bot_mean, bot_std = load_metric(exp, version, eval_name, metric_bot)
            grid[(ri, ci)] = (top_mean, top_std, bot_mean, bot_std)
            status = "ok" if np.isfinite(top_mean) else "missing"
            print(
                f"  {exp}: {metric_top}={top_mean:.3f} ±{top_std:.3f}  "
                f"{metric_bot}={bot_mean:.3f} ±{bot_std:.3f}  [{status}]"
                if np.isfinite(top_mean)
                else f"  {exp}: [{status}]"
            )
    return grid


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval name, e.g. train_KG_S")
    p.add_argument("--version", default="version_0", help="Experiment version folder")
    p.add_argument("--metric_top", default="rmse", help="Metric for upper-right triangle (default: rmse)")
    p.add_argument("--metric_bot", default="mae", help="Metric for lower-left triangle (default: mae)")
    p.add_argument("--out", default=None, help="Output path (default: results/sweep_overview_<eval>.pdf)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Loading metrics for eval={args.eval}, version={args.version} …")
    grid = build_grid(args.version, args.eval, args.metric_top, args.metric_bot)

    n_found = sum(1 for v in grid.values() if np.isfinite(v[0]))
    if n_found == 0:
        print("No metrics found. Run inference and evaluation first.", file=sys.stderr)
        sys.exit(1)
    print(f"\n{n_found}/16 experiments with data.")

    out = Path(args.out or f"results/sweep_overview_{args.eval}.pdf")
    sweep_overview(grid, args.metric_top, args.metric_bot, args.eval, out)
