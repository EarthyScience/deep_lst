"""Forecast-limit summary across datasets or classes for one model.

The forecast limit is the lead time (in days) at which the skill score
``1 - RMSE_forecast / RMSE_baseline`` first drops below zero. Skill is reduced
over space per init_time (aggregation="region_time"), then summarised into
q05/q50/q95 curves over init_time. The forecast limit is the first lead time
where the curve crosses zero: q50 -> central marker, q05 -> lower error bar
(low skill crosses earliest), q95 -> upper error bar. A curve that never
crosses within the rollout horizon is drawn right-censored (arrow at horizon);
a curve with no data, or no skill from lead 0, is drawn as a square at day 0.

Each evaluator pass runs in its own spawn subprocess so peak memory is reclaimed
between passes (the in-process loop OOMs on the full init range). Only rmse is
computed (metrics=("rmse",)) since that is all the forecast limit needs.

Two y-axis modes (``--groupby``):

dataset   One row per dataset (train sites + dense eval regions), coloured by
          baseline. Missing forecasts leave a reserved blank row.
<class>   One row per climate / land-cover class, pooled across ALL datasets,
          for a single baseline (--class_baseline, default clim_smooth). Two
          markers per row coloured by source: training vs. evaluation sites.
          <class> is any class coord present in the eval indices (kg_class,
          kg_major; modis_lc_classes only if the indices were built with it).

Usage
-----
uv run scripts/evaluation/run_fc_limits.py \\
    --experiment mlp_rollout_L_KG_mse_h8_v5_large --version version_0 \\
    --baselines era5 clim_smooth persistence

uv run scripts/evaluation/run_fc_limits.py \\
    --experiment mlp_rollout_L_KG_mse_h8_v5_large --version version_0 \\
    --groupby kg_class --class_baseline clim_smooth
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import dask
import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.evaluator import EvaluatorVerification
from deep_lst.evaluation.plots._design import CASE_COLORS
from deep_lst.evaluation.plots.fc_limits import forecast_limit_plot
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

_BASELINES = {
    "era5": "ERA5",
    "clim_hourly": "Climatology",
    "clim_smooth": "Climatology (smooth)",
    "persistence": "Persistence",
}

_EVALS = [
    "train_KG_L",
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
    "train_KG_L": "Training sites",
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

# Plot B: colour encodes training vs. evaluation locations.
_SOURCE_STYLE = {
    "train": {"color": "#d62728", "marker": "o", "label": "Training sites"},
    "eval": {"color": "#1f77b4", "marker": "o", "label": "Evaluation sites"},
}


def _source(eval_name: str) -> str:
    return "train" if eval_name.startswith("train_") else "eval"


def _lead_days(da: xr.DataArray) -> np.ndarray:
    return pd.TimedeltaIndex(da["lead_time"].values).total_seconds().to_numpy() / 86400.0


def _first_crossing(values: np.ndarray, lead_days: np.ndarray) -> float | None:
    """First lead time (days) where the skill curve drops below zero, else None."""
    below = np.asarray(values, dtype=float) < 0  # NaN -> False (not a crossing)
    idx = np.flatnonzero(below)
    return float(lead_days[idx[0]]) if idx.size else None


def _forecast_limit(skill_da: xr.DataArray) -> dict:
    """skill_da dims (lead_time, init_time): skill = 1 - RMSE_fc/RMSE_baseline.

    Returns {computable, central, lo, hi} forecast limits (days). q05 (low skill)
    crosses earliest -> lo; q95 (high skill) crosses latest -> hi. ``computable``
    is False when the median is all-NaN (missing data, distinct from a finite
    median that simply never crosses zero).
    """
    lead_days = _lead_days(skill_da)
    q = {p: skill_da.quantile(p, dim="init_time").values for p in (0.05, 0.50, 0.95)}
    return {
        "computable": bool(np.isfinite(q[0.50]).any()),
        "central": _first_crossing(q[0.50], lead_days),
        "lo": _first_crossing(q[0.05], lead_days),
        "hi": _first_crossing(q[0.95], lead_days),
    }


# --- subprocess workers (module-level so spawn can pickle them) ---


def _dataset_worker(base_kwargs: dict, region: str, baseline: str):
    """Run one verification evaluator and return its (tiny) forecast-limit dict."""
    dask.config.set(scheduler="threads", num_workers=4)  # bounded threads: fast + memory-capped
    ev = EvaluatorVerification(**base_kwargs, eval_name=region, baseline=baseline)
    ev.run()
    skill = (1 - ev.skill_scores)["rmse"]
    return _forecast_limit(skill), float(_lead_days(skill).max())


def _pool_worker(base_kwargs: dict, names: list[str], class_key: str, baseline: str):
    """Per-class forecast limits pooled across `names`, for one source.

    Reduces each region to per-class squared-error sums + valid counts and adds
    them up, so peak memory is one region's cube — not all regions at once. The
    pooled RMSE is sqrt(sum_sse / sum_count), i.e. pooling squared errors across
    all pixels of a class (not averaging per-region skills).
    """
    dask.config.set(scheduler="threads", num_workers=4)  # bounded threads: fast + memory-capped
    totals, horizon = {}, None
    for name in names:
        if not PATHS.forecast_path(base_kwargs["experiment"], base_kwargs["version"], name).exists():
            print(f"  [skip] {name}: forecast not found")
            continue
        try:
            ev = EvaluatorVerification(**base_kwargs, eval_name=name, baseline=baseline)
            ev.run()
        except FileNotFoundError as e:
            print(f"  [skip] {name} / {baseline}: {e}")
            continue
        if class_key not in ev.pred.coords:
            print(f"  [skip] {name}: no '{class_key}' coord")
            continue
        horizon = horizon or float(_lead_days(ev.pred).max())

        pred, obs, bl = ev.pred, ev.obs, ev.baseline_full
        cls = ev.pred[class_key]
        space = [d for d in ("pixel", "lat", "lon") if d in pred.dims]
        se_fc = (pred - obs) ** 2
        se_ref = (bl - obs) ** 2
        valid_fc = np.isfinite(pred) & np.isfinite(obs)
        valid_ref = np.isfinite(bl) & np.isfinite(obs)
        for c in np.unique(cls.values):
            if c == -1:
                continue
            m = cls == c
            agg = xr.Dataset(
                {
                    "sse_fc": se_fc.where(m).sum(space),
                    "n_fc": valid_fc.where(m, False).sum(space),
                    "sse_ref": se_ref.where(m).sum(space),
                    "n_ref": valid_ref.where(m, False).sum(space),
                }
            ).compute()  # force the reduction now so this region's cube is freed
            c = int(c)
            totals[c] = agg if c not in totals else totals[c] + agg
        del ev, se_fc, se_ref, valid_fc, valid_ref

    limits = {}
    for c, agg in totals.items():
        rmse_fc = np.sqrt(agg["sse_fc"] / agg["n_fc"])
        rmse_ref = np.sqrt(agg["sse_ref"] / agg["n_ref"])
        limits[c] = _forecast_limit(1 - rmse_fc / rmse_ref)  # skill (init_time, lead_time)
    return limits, horizon


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--evals", nargs="+", default=_EVALS)
    p.add_argument("--baselines", nargs="+", default=["era5", "clim_smooth", "persistence"])
    p.add_argument("--groupby", default="dataset", help="'dataset' or a class coord (e.g. kg_class).")
    p.add_argument("--class_baseline", default="clim_smooth", help="Single baseline used in class mode.")
    p.add_argument("--temporal", nargs="+", default=None)
    p.add_argument("--workers", type=int, default=2, help="Concurrent evaluator subprocesses (~50 GB each).")
    return p.parse_args()


def _base_kwargs(args):
    return dict(
        experiment=args.experiment,
        version=args.version,
        var="LST",
        temporal=args.temporal,
        aggregation="region_time",
        consistent_mask=False,
        overwrite=False,
        metrics=("rmse",),  # forecast limit only needs rmse
        compute_anomaly=False,  # skip ACC/anomaly cubes — huge memory, unused here
    )


def _pool(args):
    return ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))


def _build_dataset_rows(args, out_dir):
    """One row per dataset; series = baselines (coloured by baseline)."""
    base = _base_kwargs(args)
    present = {r: PATHS.forecast_path(args.experiment, args.version, r).exists() for r in args.evals}
    for r, ok in present.items():
        if not ok:
            print(f"  [skip] {r}: forecast not found")

    results, horizon = {}, None
    with _pool(args) as pool:
        futures = {
            (r, b): pool.submit(_dataset_worker, base, r, b) for r in args.evals if present[r] for b in args.baselines
        }
        for (r, b), fut in futures.items():
            try:
                fl, h = fut.result()
                results[(r, b)] = fl
                horizon = horizon or h
                print(f"  {r:<14} {b:<12} central={fl['central']}")
            except FileNotFoundError as e:
                print(f"  [skip] {r} / {b}: {e}")

    rows = [
        {
            "label": _REGION_DISPLAY.get(r, r),
            "series": {b: results[(r, b)] for b in args.baselines if (r, b) in results},
        }
        for r in args.evals
    ]
    styles = {
        b: {"color": CASE_COLORS.get(_BASELINES[b], "#777777"), "marker": "o", "label": _BASELINES[b]}
        for b in args.baselines
    }
    bl_str = "_".join(_BASELINES[b] for b in args.baselines)
    forecast_limit_plot(
        rows=rows,
        series_styles=styles,
        horizon_days=horizon or 8.0,
        save_path=out_dir / f"forecast_limit_datasets_vs_{bl_str}.pdf",
        title=f"Forecast limit by dataset — {args.experiment}",
        legend_title="Baseline",
    )


def _build_class_rows(args, out_dir):
    """One row per class, pooled across datasets; series = train/eval source."""
    class_key = args.groupby
    baseline = args.class_baseline
    base = _base_kwargs(args)
    sources = {
        "train": [e for e in args.evals if _source(e) == "train"],
        "eval": [e for e in args.evals if _source(e) == "eval"],
    }

    limits_by_source, horizon = {}, None
    with _pool(args) as pool:
        futures = {s: pool.submit(_pool_worker, base, names, class_key, baseline) for s, names in sources.items()}
        for s, fut in futures.items():
            print(f"Pooling {s} sites ({len(sources[s])}) ...")
            limits, h = fut.result()
            limits_by_source[s] = limits
            horizon = horizon or h

    meta = json.loads(PATHS.eval_meta_path("seviri").read_text())
    labels = meta.get(f"{class_key}_labels", {})

    rows = []
    for c in sorted(set(limits_by_source.get("train", {})) | set(limits_by_source.get("eval", {}))):
        series = {s: limits_by_source[s][c] for s in ("train", "eval") if c in limits_by_source.get(s, {})}
        rows.append({"label": labels.get(str(c), str(c)), "series": series})

    # Most-predictable classes (latest / censored eval crossing) at the top.
    def sort_key(r):
        pt = r["series"].get("eval") or r["series"].get("train") or {}
        c = pt.get("central")
        return np.inf if c is None else c

    rows = sorted(rows, key=sort_key, reverse=True)

    forecast_limit_plot(
        rows=rows,
        series_styles=_SOURCE_STYLE,
        horizon_days=horizon or 8.0,
        save_path=out_dir / f"forecast_limit_{class_key}_{baseline}.pdf",
        title=f"Forecast limit by {class_key} vs {_BASELINES.get(baseline, baseline)} — {args.experiment}",
        legend_title="Site type",
    )


def main():
    args = parse_args()
    period = "_".join(args.temporal) if args.temporal else "all"
    out_dir = PATHS.results_plots_dir(args.experiment, args.version, "across_sites") / period
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {args.experiment} / {args.version}")
    print(f"Group by   : {args.groupby}   Workers: {args.workers}")

    if args.groupby == "dataset":
        _build_dataset_rows(args, out_dir)
    else:
        _build_class_rows(args, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
