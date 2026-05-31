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
Available --reduction_order: space_then_time  time_then_space
"""

import argparse

import numpy as np

from deep_lst.evaluation.evaluator import CLIMATE_BASELINES, run_evaluators_across_sites
from deep_lst.evaluation.plots.across_sites import (
    cross_site_acc_boxplot,
    cross_site_rmse_boxplot,
    cross_site_skill_boxplot,
)
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

_BASELINES = {
    "era5": "ERA5",
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

_REDUCTION_ORDER_MAP = {
    "space_then_time": ("space", "time"),
    "time_then_space": ("time", "space"),
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
    p.add_argument("--temporal", nargs="+", default=None)
    p.add_argument("--evals", nargs="+", default=_ALL_EVALS)
    p.add_argument("--metric", default="rmse")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--violin", action="store_true")
    p.add_argument(
        "--reduction_order",
        choices=list(_REDUCTION_ORDER_MAP),
        default="space_then_time",
        help="Reduction order used when aggregation='total' (stored on the evaluator).",
    )
    return p.parse_args()


def run_across_sites():
    args = parse_args()
    reduction_order = _REDUCTION_ORDER_MAP[args.reduction_order]

    period = "_".join(args.temporal) if args.temporal else "all"
    out_dir = PATHS.results_plots_dir(args.experiment, args.version, "across_sites") / period
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment      : {args.experiment} / {args.version}")
    print(f"Baselines       : {args.baseline}   Lead time: {args.lead_time}")
    print(f"Reduction order : {args.reduction_order}")

    # aggregation="region_time" → scores have shape (valid_time,), one value per timestep.
    # All metrics come from ev.scores / ev.baseline_scores / ev.climate_scores; no compute_scores in this script.
    ev_kwargs = dict(
        experiment=args.experiment,
        version=args.version,
        var="LST",
        temporal=args.temporal,
        aggregation="region_time",
        groupby="diurnal",
        consistent_mask=False,
        overwrite=False,
        lead_time=args.lead_time,
        reduction_order=reduction_order,
    )

    model_evs = run_evaluators_across_sites(ev_kwargs, args.evals, baseline=None)
    baseline_evs = {b: run_evaluators_across_sites(ev_kwargs, args.evals, baseline=b) for b in args.baseline}

    # Proper (anomaly) ACC requires a climate baseline; pick the first one passed.
    climate_baseline = next((b for b in args.baseline if b in CLIMATE_BASELINES), None)
    acc_source_evs = baseline_evs[climate_baseline] if climate_baseline else None
    if acc_source_evs is None:
        print("  [warn] No climate baseline in --baseline; ACC will be skipped.")

    model_rmse, acc_vals, nse_vals = {}, {}, {}
    for region, evaluator in model_evs.items():
        rmse = evaluator.scores["rmse"].values
        model_rmse[region] = rmse[np.isfinite(rmse)]
        nse = evaluator.scores["nse"].values.ravel()
        if np.isfinite(nse).any():
            nse_vals[region] = nse[np.isfinite(nse)]
        if acc_source_evs is not None and region in acc_source_evs:
            cs = acc_source_evs[region].climate_scores
            if cs is not None and "acc" in cs:
                acc = cs["acc"].values.ravel()
                if np.isfinite(acc).any():
                    acc_vals[region] = acc[np.isfinite(acc)]

    baselines_rmse = {}
    for baseline, evaluator in baseline_evs.items():
        label = _BASELINES[baseline]
        baselines_rmse[label] = {
            region: ev_bl.baseline_scores["rmse"].values[np.isfinite(ev_bl.baseline_scores["rmse"].values)]
            for region, ev_bl in evaluator.items()
        }

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
