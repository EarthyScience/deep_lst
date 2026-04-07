"""Cross-experiment comparison: boxplots by sample size or stratification.

Loads pre-computed region_time metric files from multiple experiments
(produced by run_evaluation.py) and generates grouped boxplots comparing
MLP-Diag / MLP-Rollout / ERA5 across sample sizes or stratification choices.

Experiments are specified as:
  --diag   mlp_diag_S_KG_v1   mlp_diag_M_KG_v1   mlp_diag_L_KG_v1
  --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1

Group labels (x-axis) are inferred from the experiment names unless
overridden with --group_labels.

Usage
-----
# Group by sample size (S / M / L):
python scripts/evaluation/run_comparison.py \\
    --eval antananarivo \\
    --version version_0 \\
    --diag    mlp_diag_S_KG_v1    mlp_diag_M_KG_v1    mlp_diag_L_KG_v1 \\
    --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1 \\
    --group_by sample_size \\
    --metric rmse \\
    --lead_time 1h

# Group by stratification (KG / KGLC):
python scripts/evaluation/run_comparison.py \\
    --eval antananarivo \\
    --version version_0 \\
    --diag    mlp_diag_L_KG_v1    mlp_diag_L_KGLC_v1 \\
    --rollout mlp_rollout_L_KG_v1 mlp_rollout_L_KGLC_v1 \\
    --group_by stratification \\
    --metric rmse \\
    --lead_time 1h
"""
import argparse
import re
from pathlib import Path

import xarray as xr

from deep_lst.evaluation.plots.scores import score_boxplots
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# Regex patterns to extract group labels from experiment names
_SAMPLE_SIZE_RE = re.compile(r'_([SML])_')
_STRAT_RE = re.compile(r'_(KG(?:LC)?)_')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True,
                   help="Eval region name, e.g. antananarivo")
    p.add_argument("--version", default="version_0",
                   help="Version folder (same for all experiments)")
    p.add_argument("--diag", nargs="*", default=[],
                   metavar="EXPERIMENT",
                   help="MLP-Diag experiment names (in group order)")
    p.add_argument("--rollout", nargs="*", default=[],
                   metavar="EXPERIMENT",
                   help="MLP-Rollout experiment names (in group order)")
    p.add_argument("--group_by", choices=["sample_size", "stratification"],
                   default="sample_size",
                   help="How to label x-axis groups (default: sample_size)")
    p.add_argument("--group_labels", nargs="*", default=None,
                   metavar="LABEL",
                   help="Override auto-detected group labels (must match number of experiments)")
    p.add_argument("--metric", default="rmse",
                   help="Metric to plot (must exist in metric files, default: rmse)")
    p.add_argument("--lead_time", default="1h",
                   help="Lead time used in run_evaluation.py (affects which metric file is matched)")
    p.add_argument("--variable", default="LST")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Override output directory (default: results/comparison/{eval})")
    return p.parse_args()


def _infer_group_label(experiment: str, group_by: str) -> str:
    if group_by == "sample_size":
        m = _SAMPLE_SIZE_RE.search(experiment)
        return m.group(1) if m else experiment
    else:  # stratification
        m = _STRAT_RE.search(experiment)
        return m.group(1) if m else experiment


def _load_metric(experiment, version, eval_name, metric):
    path = PATHS.metrics_path(experiment, version, eval_name, aggregation="region_time")
    if not path.exists():
        raise FileNotFoundError(
            f"Metric file not found: {path}\n"
            f"Run run_evaluation.py for experiment '{experiment}' first."
        )
    ds = xr.open_dataset(path)
    if metric not in ds:
        raise KeyError(
            f"Metric '{metric}' not found in {path}. "
            f"Available: {list(ds.data_vars)}"
        )
    return ds


def run_comparison():
    args = parse_args()

    n_diag = len(args.diag)
    n_rollout = len(args.rollout)
    n_groups = max(n_diag, n_rollout)
    if n_groups == 0:
        raise ValueError("Pass at least one experiment via --diag or --rollout.")

    # --- Infer or validate group labels ---
    if args.group_labels is not None:
        if len(args.group_labels) != n_groups:
            raise ValueError(
                f"--group_labels has {len(args.group_labels)} entries but "
                f"{n_groups} experiment groups were given."
            )
        group_labels = args.group_labels
    else:
        ref_experiments = args.diag if args.diag else args.rollout
        group_labels = [_infer_group_label(e, args.group_by) for e in ref_experiments]

    print(f"Eval region : {args.eval}")
    print(f"Groups      : {group_labels}")
    print(f"Metric      : {args.metric}")
    print(f"Group by    : {args.group_by}")

    # --- Load metrics ---
    scores_by_group: dict[str, dict[str, xr.Dataset]] = {}
    for i, label in enumerate(group_labels):
        scores_by_group[label] = {}

        if i < n_diag:
            exp = args.diag[i]
            ds = _load_metric(exp, args.version, args.eval, args.metric)
            scores_by_group[label]["MLP-Diag"] = ds
            print(f"  Loaded MLP-Diag   [{label}]: {exp}")

        if i < n_rollout:
            exp = args.rollout[i]
            ds = _load_metric(exp, args.version, args.eval, args.metric)
            scores_by_group[label]["MLP-Rollout"] = ds
            print(f"  Loaded MLP-Rollout [{label}]: {exp}")

    # ERA5 baseline: load once and share across all groups
    baseline_path = PATHS.baseline_path(args.eval)
    if baseline_path.exists():
        # Build a minimal scores dataset from the baseline if available
        # (ERA5 scores would need to be pre-computed by run_evaluation.py for a rollout experiment)
        # Look for ERA5 metrics in any rollout experiment (region_time, rmse)
        era5_ds = None
        for exp in args.rollout or args.diag:
            era5_metrics = PATHS.metrics_path(exp, args.version, args.eval, "region_time")
            # ERA5 scores are stored separately — convention: results/era5_baseline/{eval}/metrics_region_time.nc
            era5_path = PATHS.results_root / "era5_baseline" / args.eval / "metrics_region_time.nc"
            if era5_path.exists():
                era5_ds = xr.open_dataset(era5_path)
                print(f"  Loaded ERA5 baseline metrics: {era5_path}")
                break
        if era5_ds is not None:
            for label in group_labels:
                scores_by_group[label]["ERA5"] = era5_ds
    else:
        print(f"  ERA5 baseline not found at {baseline_path} — skipping ERA5 comparison")

    # --- Output path ---
    xlabel = "Sample size" if args.group_by == "sample_size" else "Stratification"
    title = f"{args.eval} — {args.variable} {args.metric.upper()} by {xlabel.lower()} (lead_time={args.lead_time})"

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = PATHS.results_root / "comparison" / args.eval
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / f"{args.variable}_{args.metric}_{args.group_by}_lt{args.lead_time}.png"

    score_boxplots(
        scores_by_group=scores_by_group,
        metric=args.metric,
        xlabel=xlabel,
        save_path=save_path,
        title=title,
    )
    print(f"\nSaved comparison plot: {save_path}")


if __name__ == "__main__":
    run_comparison()
