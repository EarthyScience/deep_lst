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
    --baselines era5 clim_smooth persistence

Available --baselines:
    era5          ERA5 skin temperature
    clim_hourly   SEVIRI hourly climatology
    clim_smooth   SEVIRI smoothed climatology
    persistence   SEVIRI obs at init_time
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import dask

from deep_lst.evaluation.evaluator import EvaluatorDiagnostics, EvaluatorVerification
from deep_lst.evaluation.plot_runner import PlotRunnerDiagnostics, PlotRunnerVerification
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

BASELINE_LABELS = {
    "era5": "ERA5",
    "era5_land": "ERA5-Land",
    "clim_hourly": "hourly clim",
    "clim_smooth": "smoothed clim",
    "persistence": "persistence",
}


def _run_evaluator(cls_name, kwargs):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    dask.config.set(
        scheduler="threads", num_workers=2
    )  # bounded threads: caps memory without OOM; 8 tasks already run as separate processes
    cls = {"EvaluatorDiagnostics": EvaluatorDiagnostics, "EvaluatorVerification": EvaluatorVerification}[cls_name]
    ev = cls(**kwargs)
    ev.run()
    for attr in (
        "scores",
        "grouped",
        "pred",
        "obs",
        "baseline_scores",
        "baseline_grouped",
        "baseline_aligned",
        "skill_scores",
        "climate_scores",
        "climate_grouped",
        "pred_anom",
        "obs_anom",
    ):
        val = getattr(ev, attr, None)
        if val is not None and hasattr(val, "compute"):
            setattr(ev, attr, val.compute())
    return ev


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

    base_kwargs = dict(
        experiment=args.experiment,
        version=args.version,
        eval_name=args.eval,
        var=args.variable,
        temporal=args.temporal,
        consistent_mask=False,
        overwrite=args.overwrite,
    )
    diag_extra = dict(aggregation="region_space", groupby="diurnal", lead_time=args.lead_time)
    verif_extra = dict(aggregation="region_time")
    baselines = args.baselines or []

    task_specs = {
        ("Diag", None): ("EvaluatorDiagnostics", {**base_kwargs, **diag_extra, "baseline": None}),
        ("Verif", None): ("EvaluatorVerification", {**base_kwargs, **verif_extra, "baseline": None}),
    }
    for b in baselines:
        task_specs[("Diag", b)] = ("EvaluatorDiagnostics", {**base_kwargs, **diag_extra, "baseline": b})
        task_specs[("Verif", b)] = ("EvaluatorVerification", {**base_kwargs, **verif_extra, "baseline": b})

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(task_specs), mp_context=ctx) as pool:
        futures = {key: pool.submit(_run_evaluator, cls_name, kw) for key, (cls_name, kw) in task_specs.items()}
        results = {key: f.result() for key, f in futures.items()}

    eval_diagnostics = results[("Diag", None)]
    eval_verification = results[("Verif", None)]
    baselines_diagnostics = {BASELINE_LABELS[b]: results[("Diag", b)] for b in baselines}
    baselines_verification = {BASELINE_LABELS[b]: results[("Verif", b)] for b in baselines}

    PlotRunnerDiagnostics(eval_diagnostics, label, plots_dir, baselines_diagnostics).run(args.plots)
    PlotRunnerVerification(eval_verification, label, plots_dir, baselines_verification).run(args.plots)

    print("\nDone.")


if __name__ == "__main__":
    start = time.time()
    run_evaluation()
    print(f"Total runtime in minutes: {(time.time() - start) / 60:.1f}")
