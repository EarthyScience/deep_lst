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
uv run scripts/evaluation/run_comparison.py
    --eval dahra
    --version version_1
    --diag    mlp_diag_S_KG_v1    mlp_diag_M_KG_v1    mlp_diag_L_KG_v1
    --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1
    --group_by sample_size
    --metric rmse
    --lead_time 3h

# Group by stratification (KG / KGLC):
uv run scripts/evaluation/run_comparison.py
    --eval dahra
    --version version_1
    --diag    mlp_diag_L_KG_v1    mlp_diag_L_KGLC_v1
    --rollout mlp_rollout_L_KG_v1 mlp_rollout_L_KGLC_v1
    --group_by stratification
    --metric rmse
    --lead_time 3h
"""

import argparse
import re
from pathlib import Path

import xarray as xr

from deep_lst.evaluation.plots.diurnal import diurnal_delta_comparison
from deep_lst.evaluation.plots.scores import score_boxplots
from deep_lst.evaluation.plots.spectra import power_spectrum_from_forecasts
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# Regex patterns to extract group labels from experiment names
_SAMPLE_SIZE_RE = re.compile(r"_([SML])_")
_STRAT_RE = re.compile(r"_(KG(?:LC)?)_")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval region name, e.g. antananarivo")
    p.add_argument("--version", default="version_0", help="Version folder (same for all experiments)")
    p.add_argument(
        "--diag", nargs="*", default=[], metavar="EXPERIMENT", help="MLP-Diag experiment names (in group order)"
    )
    p.add_argument(
        "--rollout", nargs="*", default=[], metavar="EXPERIMENT", help="MLP-Rollout experiment names (in group order)"
    )
    p.add_argument(
        "--group_by",
        choices=["sample_size", "stratification"],
        default="sample_size",
        help="How to label x-axis groups (default: sample_size)",
    )
    p.add_argument(
        "--group_labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Override auto-detected group labels (must match number of experiments)",
    )
    p.add_argument("--metric", default="rmse", help="Metric to plot (must exist in metric files, default: rmse)")
    p.add_argument(
        "--lead_time", default="1h", help="Lead time used in run_evaluation.py (affects which metric file is matched)"
    )
    p.add_argument("--variable", default="LST")
    p.add_argument(
        "--plots",
        nargs="+",
        default=["boxplot", "diurnal_delta", "power_spectrum"],
        metavar="PLOT",
        help="Plots to generate. Choose from: boxplot diurnal_delta power_spectrum (default: all)",
    )
    p.add_argument(
        "--output_dir", type=Path, default=None, help="Override output directory (default: results/comparison/{eval})"
    )
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
            f"Metric file not found: {path}\nRun run_evaluation.py for experiment '{experiment}' first."
        )
    ds = xr.open_dataset(path)
    if metric not in ds:
        raise KeyError(f"Metric '{metric}' not found in {path}. Available: {list(ds.data_vars)}")
    return ds


def _load_diurnal_mean(experiment, version, eval_name):
    path = PATHS.metrics_path(experiment, version, eval_name, aggregation="diurnal_mean")
    if not path.exists():
        raise FileNotFoundError(
            f"Diurnal mean file not found: {path}\nRun run_evaluation.py for experiment '{experiment}' first."
        )
    return xr.open_dataset(path)


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
                f"--group_labels has {len(args.group_labels)} entries but {n_groups} experiment groups were given."
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

    requested_plots = set(args.plots)

    if "boxplot" in requested_plots:
        score_boxplots(
            scores_by_group=scores_by_group,
            metric=args.metric,
            xlabel=xlabel,
            save_path=save_path,
            title=title,
        )
        print(f"Saved: {save_path}")

    if "diurnal_delta" in requested_plots:
        # Load diurnal_mean for the first diag + first rollout experiment only
        # (one representative per model type, typically the L experiment)
        diurnal_by_case = {}
        if args.diag:
            try:
                diurnal_by_case["MLP-Diag"] = _load_diurnal_mean(args.diag[-1], args.version, args.eval)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        if args.rollout:
            try:
                diurnal_by_case["MLP-Rollout"] = _load_diurnal_mean(args.rollout[-1], args.version, args.eval)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        if diurnal_by_case:
            dd_path = out_dir / f"{args.variable}_diurnal_delta_comparison_lt{args.lead_time}.png"
            diurnal_delta_comparison(
                diurnal_mean_by_case=diurnal_by_case,
                var=args.variable,
                save_path=dd_path,
                eval_region=args.eval,
            )
            print(f"Saved: {dd_path}")

    if "power_spectrum" in requested_plots:
        # Load forecast files for the last (largest / most representative) experiment
        # per model type and compute spatial power spectra.
        forecast_by_case = {}
        for case_label, experiments in [("MLP-Diag", args.diag), ("MLP-Rollout", args.rollout)]:
            if not experiments:
                continue
            exp = experiments[-1]
            fc_path = PATHS.forecast_path(exp, args.version, args.eval)
            if not fc_path.exists():
                print(f"Warning: forecast not found for {exp} — skipping power spectrum")
                continue
            forecast_by_case[case_label] = xr.open_zarr(fc_path)

        if forecast_by_case:
            ps_path = out_dir / f"{args.variable}_power_spectrum_lt{args.lead_time}.png"
            power_spectrum_from_forecasts(
                forecast_by_case=forecast_by_case,
                var=args.variable,
                lead_time=args.lead_time,
                save_path=ps_path,
                eval_region=args.eval,
            )
            for ds in forecast_by_case.values():
                ds.close()


if __name__ == "__main__":
    run_comparison()
