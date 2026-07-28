"""Cross-experiment comparison: boxplots by sample size or stratification.

Compares MLP-Diag / MLP-Rollout variants across groups (sample size or
stratification). Experiments are specified as:
  --diag   mlp_diag_S_KG_v1   mlp_diag_M_KG_v1   mlp_diag_L_KG_v1
  --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1

Group labels (x-axis) are inferred from experiment names unless overridden.

Usage
-----
uv run scripts/evaluation/run_comparison.py \\
    --eval dahra \\
    --version version_1 \\
    --diag    mlp_diag_S_KG_v1    mlp_diag_M_KG_v1    mlp_diag_L_KG_v1 \\
    --rollout mlp_rollout_S_KG_v1 mlp_rollout_M_KG_v1 mlp_rollout_L_KG_v1 \\
    --group_by sample_size --metric rmse --lead_time 3h
"""

import argparse
import re
from pathlib import Path

import xarray as xr

from deep_lst.evaluation.evaluator import EvaluatorVerification
from deep_lst.evaluation.plot_runner import _quantile_bands, _with_delta
from deep_lst.evaluation.plots.diurnal import diurnal_delta_comparison
from deep_lst.evaluation.plots.scores import score_boxplots
from deep_lst.evaluation.plots.spectra import power_spectrum_from_forecasts
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

_SAMPLE_SIZE_RE = re.compile(r"_([SML])_")
_STRAT_RE = re.compile(r"_(KG(?:LC)?)_")


def _infer_group_label(experiment: str, group_by: str) -> str:
    if group_by == "sample_size":
        m = _SAMPLE_SIZE_RE.search(experiment)
        return m.group(1) if m else experiment
    m = _STRAT_RE.search(experiment)
    return m.group(1) if m else experiment


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True)
    p.add_argument("--version", default="version_0")
    p.add_argument("--diag", nargs="*", default=[])
    p.add_argument("--rollout", nargs="*", default=[])
    p.add_argument("--group_by", choices=["sample_size", "stratification"], default="sample_size")
    p.add_argument("--group_labels", nargs="*", default=None)
    p.add_argument("--metric", default="rmse")
    p.add_argument("--lead_time", default="1h")
    p.add_argument("--temporal", nargs="+", default=None)
    p.add_argument("--variable", default="LST")
    p.add_argument("--plots", nargs="+", default=["boxplot", "diurnal_delta", "power_spectrum"])
    p.add_argument("--output_dir", type=Path, default=None)
    return p.parse_args()


def _make_products(experiment, version, eval_name, lead_time, temporal) -> dict:
    """One region's diagnostic products for the comparison: per-timestep scores + diurnal bands."""
    ev = EvaluatorVerification(
        experiment=experiment,
        version=version,
        eval_name=eval_name,
        var="LST",
        baseline=None,
        temporal=temporal,
        aggregation="region_time",
        consistent_mask=False,
        overwrite=False,
    )
    ev.load()
    ev.compute_diagnostics(lead_time, ["diag_scores_by_time", "diag_by_hour"])
    ev.materialise_diagnostics()
    return ev.diag_products


def run_comparison():
    args = parse_args()
    n_groups = max(len(args.diag), len(args.rollout))
    if n_groups == 0:
        raise ValueError("Pass at least one experiment via --diag or --rollout.")

    ref = args.diag if args.diag else args.rollout
    group_labels = args.group_labels or [_infer_group_label(e, args.group_by) for e in ref]

    out_dir = args.output_dir or (PATHS.results_root / "comparison" / args.eval)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.plots)

    print(f"Eval region : {args.eval}")
    print(f"Groups      : {group_labels}")
    print(f"Metric      : {args.metric}   Lead time: {args.lead_time}")

    # Load one evaluator per (case_type, group)
    ev_by: dict[tuple[str, str], dict] = {}
    for case_type, experiments in [("MLP-Diag", args.diag), ("MLP-Rollout", args.rollout)]:
        for i, exp in enumerate(experiments):
            try:
                ev_by[(case_type, group_labels[i])] = _make_products(
                    exp, args.version, args.eval, args.lead_time, args.temporal
                )
                print(f"  [ok] {case_type} [{group_labels[i]}]: {exp}")
            except FileNotFoundError as e:
                print(f"  [skip] {exp}: {e}")

    if not ev_by:
        print("No forecasts loaded. Exiting.")
        return

    if "boxplot" in requested:
        scores_by_group: dict[str, dict[str, xr.Dataset]] = {}
        for (case_type, group_label), prods in ev_by.items():
            scores_by_group.setdefault(group_label, {})[case_type] = prods["diag_scores_by_time"]
        xlabel = "Sample size" if args.group_by == "sample_size" else "Stratification"
        score_boxplots(
            scores_by_group=scores_by_group,
            metric=args.metric,
            xlabel=xlabel,
            save_path=out_dir / f"{args.variable}_{args.metric}_{args.group_by}_lt{args.lead_time}.png",
            title=f"{args.eval} — {args.variable} {args.metric.upper()} by {xlabel.lower()} (lead_time={args.lead_time})",
        )

    if "diurnal_delta" in requested:
        diurnal_by_case = {}
        for case_type in ("MLP-Diag", "MLP-Rollout"):
            keys = [k for k in ev_by if k[0] == case_type]
            if not keys:
                continue
            prods = ev_by[keys[-1]]  # last = most representative experiment
            by_hour = prods["diag_by_hour"]
            space_dims = [d for d in by_hour.dims if d in ("lat", "lon", "pixel")]
            diurnal_by_case[case_type] = _with_delta(_quantile_bands(by_hour, space_dims))
        if diurnal_by_case:
            diurnal_delta_comparison(
                diurnal_mean_by_case=diurnal_by_case,
                var=args.variable,
                save_path=out_dir / f"{args.variable}_diurnal_delta_comparison_lt{args.lead_time}.png",
                eval_region=args.eval,
            )

    if "power_spectrum" in requested:
        forecast_by_case = {}
        for case_type, experiments in [("MLP-Diag", args.diag), ("MLP-Rollout", args.rollout)]:
            if not experiments:
                continue
            fc_path = PATHS.forecast_path(experiments[-1], args.version, args.eval)
            if not fc_path.exists():
                print(f"  [warn] forecast not found for {experiments[-1]}")
                continue
            forecast_by_case[case_type] = xr.open_zarr(fc_path)
        if forecast_by_case:
            power_spectrum_from_forecasts(
                forecast_by_case=forecast_by_case,
                var=args.variable,
                lead_time=args.lead_time,
                save_path=out_dir / f"{args.variable}_power_spectrum_lt{args.lead_time}.png",
                eval_region=args.eval,
            )
            for ds in forecast_by_case.values():
                ds.close()

    print("\nDone.")


if __name__ == "__main__":
    run_comparison()
