"""Across-sites evaluation: summarise one model's performance over all eval regions.

Runs the evaluator for each region on-the-fly and produces summary plots with
all sites on a shared x-axis.

Plots generated
---------------
cross_site_rmse_boxplot   Paired boxplots: model + baselines per region
cross_site_acc_boxplot    Per-pixel ACC distribution per region

Usage
-----
uv run scripts/evaluation/run_across_sites.py \\
    --experiment mlp_rollout_L_KG_v1 \\
    --version version_1 \\
    --baseline era5 clim_smooth \\
    --lead_time 3h

Available --baseline:  era5  clim_hourly  clim_smooth  persistence
"""

import argparse

import numpy as np

from deep_lst.evaluation.evaluator import EvaluatorDiagnostics
from deep_lst.evaluation.metrics import compute_scores
from deep_lst.evaluation.plots.across_sites import cross_site_acc_boxplot, cross_site_rmse_boxplot
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

BASELINE_LABELS = {
    "era5": "ERA5",
    "clim_hourly": "hourly clim",
    "clim_smooth": "smoothed clim",
    "persistence": "persistence",
}

_BASELINE_PLOT_LABEL = {
    "ERA5": "ERA5",
    "hourly clim": "Climatology",
    "smoothed clim": "Climatology (smooth)",
    "persistence": "Persistence",
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

_REGION_DISPLAY = {
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--baseline", nargs="+", default=["era5"])
    p.add_argument("--lead_time", default="3h")
    p.add_argument("--evals", nargs="+", default=_ALL_EVALS)
    p.add_argument("--metric", default="rmse")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--violin", action="store_true")
    return p.parse_args()


def run_across_sites():
    args = parse_args()
    out_dir = PATHS.results_dir(args.experiment, args.version, "across_sites")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Baselines  : {args.baseline}   Lead time: {args.lead_time}")

    model_vals: dict[str, np.ndarray] = {}
    acc_vals: dict[str, np.ndarray] = {}
    baselines_vals: dict[str, dict[str, np.ndarray]] = {
        BASELINE_LABELS[b]: {} for b in args.baseline if b in BASELINE_LABELS
    }

    ev_kwargs = dict(
        experiment=args.experiment,
        version=args.version,
        var="LST",
        metrics=["rmse"],
        temporal=None,
        aggregation="region_space",
        groupby="diurnal",
        consistent_mask=False,
        overwrite=False,
        lead_time=args.lead_time,
    )

    for eval_region in args.evals:
        try:
            ev = EvaluatorDiagnostics(**ev_kwargs, eval_name=eval_region, baseline=None)
            ev.run()
        except FileNotFoundError:
            print(f"  [skip] {eval_region}: forecast not found")
            continue

        rmse = compute_scores(ev.pred, ev.obs, ["rmse"], "region_time")["rmse"].values
        model_vals[eval_region] = rmse[np.isfinite(rmse)]

        acc_flat = compute_scores(ev.pred, ev.obs, ["anomaly_correlation"], "region_space")[
            "anomaly_correlation"
        ].values.ravel()
        if np.isfinite(acc_flat).any():
            acc_vals[eval_region] = acc_flat[np.isfinite(acc_flat)]

        print(f"  [ok] {eval_region}: {len(model_vals[eval_region])} timesteps")

        for b_key in args.baseline:
            if b_key not in BASELINE_LABELS:
                print(f"  [warn] unknown baseline {b_key!r}")
                continue
            try:
                ev_bl = EvaluatorDiagnostics(**ev_kwargs, eval_name=eval_region, baseline=b_key)
                ev_bl.run()
            except FileNotFoundError:
                print(f"  [skip] {eval_region}/{b_key}: baseline not found")
                continue
            bl_rmse = compute_scores(ev_bl.baseline_aligned, ev_bl.obs, ["rmse"], "region_time")["rmse"].values
            baselines_vals[BASELINE_LABELS[b_key]][eval_region] = bl_rmse[np.isfinite(bl_rmse)]

    if not model_vals:
        print("No model metrics found. Exiting.")
        return

    model_vals = {_REGION_DISPLAY.get(r, r): v for r, v in model_vals.items()}
    acc_vals = {_REGION_DISPLAY.get(r, r): v for r, v in acc_vals.items()}
    baselines_vals = {
        _BASELINE_PLOT_LABEL.get(bl, bl): {_REGION_DISPLAY.get(r, r): v for r, v in rv.items()}
        for bl, rv in baselines_vals.items()
    }

    bl_str = "_".join(BASELINE_LABELS.get(b, b) for b in args.baseline)
    cross_site_rmse_boxplot(
        model_vals_by_region=model_vals,
        baselines_vals_by_region=baselines_vals,
        model_label="Deep LST",
        metric=args.metric,
        save_path=out_dir / f"cross_site_{args.metric}_lt{args.lead_time}_vs_{bl_str}.pdf",
        title=f"Deep LST — {args.metric.upper()} across sites (lt={args.lead_time})",
        violin=args.violin,
    )

    if acc_vals:
        cross_site_acc_boxplot(
            acc_vals_by_region=acc_vals,
            save_path=out_dir / f"cross_site_acc_lt{args.lead_time}.pdf",
            violin=args.violin,
        )
    else:
        print("  [skip] ACC plot: no ACC values computed (grid data required)")

    print("\nDone.")


if __name__ == "__main__":
    run_across_sites()
