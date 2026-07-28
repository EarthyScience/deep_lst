"""Across-sites evaluation: summarise one model's performance over all eval regions.

Runs the evaluator for each region on-the-fly and produces summary plots with
all sites on a shared x-axis.

Plots generated
---------------
cross_site_rmse_boxplot   Paired boxplots: model + baselines per region
cross_site_acc_boxplot    Per-timestep ACC distribution per region
cross_site_skill_boxplot  Side-by-side ACC and NSE per region

Usage
-----
uv run scripts/evaluation/run_across_sites.py \
    --experiment mlp_rollout_L_KG_mse_h24_v5_ft \
    --version version_0 \
    --baseline era5 clim_smooth \
    --lead_time 6h

Available --baseline:        era5  clim_hourly  clim_smooth  persistence
"""

import argparse

import numpy as np

from deep_lst.evaluation.evaluator import CLIMATE_BASELINES, Evaluator
from deep_lst.evaluation.plots.across_sites import (
    cross_site_acc_boxplot,
    cross_site_rmse_boxplot,
    cross_site_skill_boxplot,
)
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

_BASELINES = {
    "era5": "ERA5",
    "era5_land": "ERA5-Land",
    "clim_hourly": "Climatology",
    "clim_smooth": "Climatology (smooth)",
    "persistence": "Persistence",
}

_ALL_EVALS = [
    "dahra",
    "antananarivo",
    "bouamir",
    "gobabeb",
    "johannesburg",
    "passage",
    "sahel_niger",
    "west_africa",
    "yangambi",
]

_REGION_DISPLAY = {
    "dahra": "Dahra",
    "antananarivo": "Antananarivo",
    "bouamir": "Bouamir",
    "gobabeb": "Gobabeb",
    "johannesburg": "Johannesburg",
    "passage": "Passage",
    "sahel_niger": "Sahel",
    "west_africa": "West Africa",
    "yangambi": "Yangambi",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--baseline", nargs="+", default=["era5_land"])
    p.add_argument("--lead_time", default="3h")
    p.add_argument("--temporal", nargs="+", default=None)
    p.add_argument("--evals", nargs="+", default=_ALL_EVALS)
    p.add_argument("--metric", default="rmse")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--violin", action="store_true")
    return p.parse_args()


def run_across_sites():
    args = parse_args()
    period = "_".join(args.temporal) if args.temporal else "all"
    out_dir = PATHS.results_plots_dir(args.experiment, args.version, "across_sites") / period
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment      : {args.experiment} / {args.version}")
    print(f"Baselines       : {args.baseline}   Lead time: {args.lead_time}")

    # Per-timestep (region_time) scores per region, from two diagnostic products:
    # diag_scores_by_time carries the model + per-baseline columns, diag_anomaly_by_time the ACC.
    climate_baseline = next((b for b in args.baseline if b in CLIMATE_BASELINES), None)
    if climate_baseline is None:
        print("  [warn] No climate baseline in --baseline; ACC will be skipped.")
    products = ["diag_scores_by_time"] + (["diag_anomaly_by_time"] if climate_baseline else [])

    region_products = {}
    for region in args.evals:
        ev = Evaluator(
            experiment=args.experiment,
            version=args.version,
            eval_name=region,
            var="LST",
            temporal=args.temporal,
        )
        try:
            ev.load()
            ev.compute_diagnostics(
                args.lead_time, products, baselines=tuple(args.baseline), clim_baseline=climate_baseline
            )
            ev.materialise_diagnostics()
        except FileNotFoundError as error:
            print(f"  [skip] {region}: {error}")
            continue
        region_products[region] = ev.diag_products
        print(f"  [ok] {region}")

    model_rmse, acc_vals, nse_vals = {}, {}, {}
    baselines_rmse = {_BASELINES.get(b, b): {} for b in args.baseline}
    for region, prods in region_products.items():
        by_time = prods["diag_scores_by_time"]
        rmse = by_time["rmse"].values
        model_rmse[region] = rmse[np.isfinite(rmse)]
        nse = by_time["nse"].values.ravel()
        if np.isfinite(nse).any():
            nse_vals[region] = nse[np.isfinite(nse)]
        for baseline in args.baseline:
            column = f"rmse_{baseline}"
            if column in by_time:
                values = by_time[column].values
                baselines_rmse[_BASELINES.get(baseline, baseline)][region] = values[np.isfinite(values)]
        anomaly = prods.get("diag_anomaly_by_time")
        if anomaly is not None and "acc" in anomaly:
            acc = anomaly["acc"].values.ravel()
            if np.isfinite(acc).any():
                acc_vals[region] = acc[np.isfinite(acc)]

    # --- apply display names ---
    model_rmse = {_REGION_DISPLAY.get(r, r): v for r, v in model_rmse.items()}
    acc_vals = {_REGION_DISPLAY.get(r, r): v for r, v in acc_vals.items()}
    nse_vals = {_REGION_DISPLAY.get(r, r): v for r, v in nse_vals.items()}
    baselines_rmse = {bl: {_REGION_DISPLAY.get(r, r): v for r, v in rv.items()} for bl, rv in baselines_rmse.items()}

    bl_str = "_".join(_BASELINES.get(b, b) for b in args.baseline)
    cross_site_rmse_boxplot(
        model_vals_by_region=model_rmse,
        baselines_vals_by_region=baselines_rmse,
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
        print("  [skip] ACC plot: no finite ACC values")

    if acc_vals or nse_vals:
        cross_site_skill_boxplot(
            acc_vals_by_region=acc_vals,
            nse_vals_by_region=nse_vals,
            save_path=out_dir / f"cross_site_skill_lt{args.lead_time}.pdf",
            violin=args.violin,
        )
    else:
        print("  [skip] skill plot: no finite ACC/NSE values")

    print("\nDone.")


if __name__ == "__main__":
    run_across_sites()
