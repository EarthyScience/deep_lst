"""Per-experiment evaluation: compute metrics and generate all plots.

Benchmark plots are skipped automatically when no --baselines are provided.

Usage
-----
uv run scripts/evaluation/run_evaluation.py
    --experiment mlp_rollout_L_KG_crps_lb6_bptt_v3
    --version version_0
    --eval sahel_niger
    --lead_time 6h
    --temporal test_period_23
    --baselines era5_land clim_smooth persistence

Available --baselines:
    era5_land     ERA5-Land skin temperature (the reference reanalysis baseline)
    clim_hourly   SEVIRI hourly climatology
    clim_smooth   SEVIRI smoothed climatology
    persistence   SEVIRI obs at init_time
    era5          UNAVAILABLE: the era5 store carries no skt, so the files cannot be
                  re-extracted; the existing ones are stale (float32 coords, pre-fix
                  extraction). Do not use for paper results.
"""

import argparse
import time

import dask

from deep_lst.evaluation.evaluator import CLIMATE_BASELINES, EvaluatorVerification
from deep_lst.evaluation.plot_runner import PlotRunnerDiagnostics, PlotRunnerVerification, required_products
from deep_lst.evaluation.products import DIAG_PRODUCTS, PRODUCTS
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

BASELINE_LABELS = {
    "era5": "ERA5",
    "era5_land": "ERA5-Land",
    "clim_hourly": "hourly clim",
    "clim_smooth": "smoothed clim",
    "persistence": "persistence",
}


def _detect_case_label(experiment: str) -> str:
    if "rollout" in experiment.lower():
        return "MLP-Rollout"
    if "diag" in experiment.lower():
        return "MLP-Diag"
    return experiment


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--variable", default="LST")
    p.add_argument("--lead_time", default="1h")
    p.add_argument("--baselines", nargs="*", default=None)
    p.add_argument("--temporal", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plots", nargs="+", default=None)
    return p.parse_args()


def run_evaluation():
    args = parse_args()
    label = _detect_case_label(args.experiment)
    period = "_".join(args.temporal) if args.temporal else "all"
    plots_dir = PATHS.results_plots_dir(args.experiment, args.version, args.eval) / period
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Eval region: {args.eval}   Variable: {args.variable}   Lead time: {args.lead_time}")
    print(f"Case label : {label}")
    print(f"Baselines  : {args.baselines or '(none)'}")

    dask.config.set(scheduler="threads", num_workers=8)  # bounded: one process runs everything now
    baselines = tuple(args.baselines or ())
    clim_baseline = next((b for b in baselines if b in CLIMATE_BASELINES), None)
    baseline_labels = {b: BASELINE_LABELS[b] for b in baselines}

    ev = EvaluatorVerification(
        experiment=args.experiment,
        version=args.version,
        eval_name=args.eval,
        var=args.variable,
        baseline=None,
        temporal=args.temporal,
        aggregation="region_time",
        consistent_mask=False,
        overwrite=args.overwrite,
    )
    ev.load()

    verification_products = list(required_products(PlotRunnerVerification, args.plots))
    if clim_baseline is None:
        verification_products = [p for p in verification_products if PRODUCTS[p].table != "anomaly_profile"]
    if not baselines:
        verification_products = [
            p for p in verification_products if not PRODUCTS[p].with_skill or p == "scores_by_init_lead"
        ]
    if verification_products:
        ev.compute(verification_products, baselines=baselines, clim_baseline=clim_baseline)
        ev.materialise()

    diagnostic_products = list(required_products(PlotRunnerDiagnostics, args.plots))
    if clim_baseline is None:
        diagnostic_products = [p for p in diagnostic_products if not DIAG_PRODUCTS[p].anomaly]
    if diagnostic_products:
        ev.compute_diagnostics(
            args.lead_time,
            diagnostic_products,
            baselines=baselines,
            clim_baseline=clim_baseline,
            class_coords=PlotRunnerDiagnostics.CLASS_COORDS,
        )
        ev.materialise_diagnostics()

    diag_runner = PlotRunnerDiagnostics(
        ev, label, plots_dir, baseline_labels, clim_baseline=clim_baseline, lead_time=args.lead_time
    )
    diag_runner.run(args.plots)
    PlotRunnerVerification(ev, label, plots_dir, baseline_labels, clim_baseline=clim_baseline).run(args.plots)

    print("\nDone.")


if __name__ == "__main__":
    start = time.time()
    run_evaluation()
    print(f"Total runtime in minutes: {(time.time() - start) / 60:.1f}")
