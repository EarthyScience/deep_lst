"""Per-experiment evaluation: compute metrics and generate plots.

Loads a saved forecast (from run_inference.py) and the ERA5 baseline
(from extract_era5_baseline.py), computes metrics for all aggregation
scenarios, and generates the requested plots.

For cross-experiment comparison plots (boxplots by sample size or
stratification), use run_comparison.py after running this script for
all experiments of interest.

Usage
-----
python scripts/evaluation/run_evaluation.py \\
    --experiment mlp_diag_L_KG_v1 \\
    --version version_0 \\
    --eval antananarivo \\
    --lead_time 3h \\
    --metrics rmse bias \\
    --plots diurnal bias_maps improvement_map

Available --plots:
    diurnal         Diurnal cycle line plot (requires ERA5 baseline for all 3 cases)
    bias_maps       Three-panel spatial bias map (A / B / C)
    improvement_map Single-panel RMSE improvement of model over ERA5

Note: bias_maps and improvement_map are only produced when an ERA5 baseline
exists. diurnal is produced for the single experiment case (no ERA5 comparison)
if --no_baseline is passed.
"""

import argparse

import xarray as xr

from deep_lst.evaluation.metrics import (
    compute_diurnal_scores,
    compute_scores,
    compute_validity_report,
    to_valid_time,
)
from deep_lst.evaluation.plots.diurnal import diurnal_line
from deep_lst.evaluation.plots.maps import bias_maps, improvement_map
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

ALL_PLOTS = ["diurnal", "bias_maps", "improvement_map"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True, help="Experiment name, e.g. mlp_diag_L_KG_v1")
    p.add_argument("--version", required=True, help="Version folder, e.g. version_0")
    p.add_argument("--eval", required=True, help="Eval region name (without eval_ prefix), e.g. antananarivo")
    p.add_argument("--variable", default="LST", help="Variable to evaluate (default: LST)")
    p.add_argument(
        "--lead_time", default="1h", help="Lead time to use for valid_time conversion and plots, e.g. 3h (default: 1h)"
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["rmse"],
        help="Metrics to compute. Choose from: rmse bias mae r2 (default: rmse)",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        default=ALL_PLOTS,
        metavar="PLOT",
        help=f"Plots to generate. Choose from: {ALL_PLOTS} (default: all)",
    )
    p.add_argument(
        "--no_baseline", action="store_true", help="Skip ERA5 baseline loading (only model metrics/diurnal produced)"
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing metric files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_forecast(experiment, version, eval_name):
    path = PATHS.forecast_path(experiment, version, eval_name)
    if not path.exists():
        raise FileNotFoundError(f"Forecast not found: {path}\nRun scripts/inference/run_inference.py first.")
    return xr.open_dataset(path)


def _load_baseline(eval_name):
    path = PATHS.baseline_path(eval_name)
    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 baseline not found: {path}\nRun scripts/evaluation/extract_era5_baseline.py first."
        )
    return xr.open_dataset(path)


def _detect_case_label(experiment: str) -> str:
    if "rollout" in experiment.lower():
        return "MLP-Rollout"
    if "diag" in experiment.lower():
        return "MLP-Diag"
    return experiment


def _save_metrics(ds_metrics, experiment, version, eval_name, aggregation, overwrite):
    path = PATHS.metrics_path(experiment, version, eval_name, aggregation)
    if path.exists() and not overwrite:
        print(f"  Metrics ({aggregation}) already exist: {path}  (--overwrite to recompute)")
        return
    ds_metrics.to_netcdf(path)
    print(f"  Saved metrics ({aggregation}): {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_evaluation():
    args = parse_args()
    var = args.variable
    case_label = _detect_case_label(args.experiment)
    plots_dir = PATHS.results_plots_dir(args.experiment, args.version, args.eval)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Eval region: {args.eval}")
    print(f"Variable   : {var}    Lead time: {args.lead_time}")
    print(f"Case       : {case_label}")

    # --- Load forecast ---
    ds = _load_forecast(args.experiment, args.version, args.eval)
    print(
        f"\nForecast dims: init_time={ds.sizes['init_time']}, "
        f"lead_time={ds.sizes['lead_time']}, "
        f"lat={ds.sizes['lat']}, lon={ds.sizes['lon']}"
    )

    # --- Convert to valid_time space ---
    pred, obs = to_valid_time(ds, var, lead_time=args.lead_time)

    # --- Validity report ---
    report = compute_validity_report(pred, obs)
    print(f"\nValidity report ({var} @ lead_time={args.lead_time}):")
    for k, v in report.items():
        pct = 100 * v / report["n_total"] if report["n_total"] else 0
        print(f"  {k:<22}: {v:>8,}  ({pct:.1f}%)")

    # --- Compute and save metrics (all aggregations) ---
    print("\nComputing metrics...")
    for agg in ["total", "region_time", "region_space"]:
        scores = compute_scores(pred, obs, metrics=args.metrics, aggregation=agg)
        _save_metrics(scores, args.experiment, args.version, args.eval, agg, args.overwrite)
        if agg == "total":
            print("  Total scores:")
            for m in args.metrics:
                val = float(scores[m].values)
                print(f"    {m.upper():<8}: {val:.4f} K")

    # Diurnal metrics
    diurnal_ds = compute_diurnal_scores(pred, obs, metrics=args.metrics)
    _save_metrics(diurnal_ds, args.experiment, args.version, args.eval, "diurnal", args.overwrite)

    # --- ERA5 baseline ---
    era5_pred = None
    if not args.no_baseline:
        try:
            ds_baseline = _load_baseline(args.eval)
            # Align ERA5 valid_time to model valid_time
            era5_pred = ds_baseline["skt"].sel(valid_time=pred["valid_time"], method="nearest")
            print(f"\nERA5 baseline loaded: {len(era5_pred.valid_time)} timesteps")
        except FileNotFoundError as e:
            print(f"\nWarning: {e}\nSkipping ERA5 comparison plots.")

    # --- Plots ---
    requested_plots = set(args.plots)
    print(f"\nGenerating plots → {plots_dir}")

    if "diurnal" in requested_plots:
        diurnal_by_case = {case_label: diurnal_ds}
        if era5_pred is not None:
            era5_diurnal = compute_diurnal_scores(era5_pred, obs, metrics=args.metrics)
            diurnal_by_case["ERA5"] = era5_diurnal
        for m in args.metrics:
            diurnal_line(
                diurnal_by_case,
                metric=m,
                save_path=plots_dir / f"{var}_diurnal_{m}_lt{args.lead_time}.png",
                title=f"{args.eval} — {var} diurnal {m.upper()} (lead_time={args.lead_time})",
            )

    if ("bias_maps" in requested_plots or "improvement_map" in requested_plots) and era5_pred is not None:
        # region_space aggregation for maps
        bias_model = compute_scores(pred, obs, metrics=["bias"], aggregation="region_space")["bias"]
        rmse_model = compute_scores(pred, obs, metrics=["rmse"], aggregation="region_space")["rmse"]

        bias_era5 = compute_scores(era5_pred, obs, metrics=["bias"], aggregation="region_space")["bias"]
        rmse_era5 = compute_scores(era5_pred, obs, metrics=["rmse"], aggregation="region_space")["rmse"]

        if "bias_maps" in requested_plots:
            bias_maps(
                bias_by_case={case_label: bias_model, "ERA5": bias_era5},
                save_path=plots_dir / f"{var}_bias_maps_lt{args.lead_time}.png",
                eval_region=args.eval,
            )

        if "improvement_map" in requested_plots:
            improvement_map(
                rmse_model=rmse_model,
                rmse_baseline=rmse_era5,
                save_path=plots_dir / f"{var}_improvement_map_lt{args.lead_time}.png",
                model_label=case_label,
                eval_region=args.eval,
            )

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    run_evaluation()
