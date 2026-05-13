"""Per-experiment evaluation: compute metrics and generate all plots.

Benchmark plots are skipped automatically when no --baselines are provided.

Usage
-----
uv run scripts/evaluation/run_evaluation.py     \
    --experiment mlp_rollout_L_KG_crps_lb6_bptt_v3 \\
    --version version_0     \
    --eval sahel_niger     \
    --lead_time 6h     \
    --temporal test_period_23 \\
    --baselines era5 clim_smooth persistence

Available --baselines:
    era5          ERA5 skin temperature
    clim_hourly   SEVIRI hourly climatology
    clim_smooth   SEVIRI smoothed climatology
    persistence   SEVIRI obs at init_time
"""

import argparse

from deep_lst.evaluation.evaluator import EvaluatorDiagnostics, EvaluatorVerification
from deep_lst.evaluation.plot_runner import PlotRunnerDiagnostics, PlotRunnerVerification
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

BASELINE_LABELS = {
    "era5": "ERA5",
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
    p.add_argument("--metrics", nargs="+", default=["rmse"])
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

    evaluator_arguments = dict(
        experiment=args.experiment,
        version=args.version,
        eval_name=args.eval,
        var=args.variable,
        metrics=args.metrics,
        temporal=args.temporal,
        consistent_mask=False,
        overwrite=args.overwrite,
    )

    def make_diag(baseline):
        eval_diagnostics = EvaluatorDiagnostics(
            **evaluator_arguments,
            baseline=baseline,
            aggregation="region_space",
            groupby="diurnal",
            lead_time=args.lead_time,
        )
        eval_diagnostics.run()
        return eval_diagnostics

    def make_verif(baseline):
        eval_verification = EvaluatorVerification(**evaluator_arguments, baseline=baseline, aggregation="region_time")
        eval_verification.run()
        return eval_verification

    eval_diagnostics = make_diag(None)
    eval_verification = make_verif(None)

    baselines_diagnostics = {BASELINE_LABELS[baseline]: make_diag(baseline) for baseline in (args.baselines or [])}
    baselines_verification = {BASELINE_LABELS[baseline]: make_verif(baseline) for baseline in (args.baselines or [])}

    PlotRunnerDiagnostics(eval_diagnostics, label, plots_dir, baselines_diagnostics).run(args.plots)
    PlotRunnerVerification(eval_verification, label, plots_dir, baselines_verification).run(args.plots)

    print("\nDone.")


if __name__ == "__main__":
    run_evaluation()
