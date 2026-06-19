"""Score tables across regions and temporal scales, in LaTeX-ready form.

Design goal: flat and readable, so every cell traces to one small function.

Rows of a table  = holdout regions (plus the "temporal" holdout).
Columns          = temporal scales: validity | hourly | daytime | nighttime
                   | amplitude | daily | total.
Lead windows     = "nowcast" (lead 1-24 h) and "full" (lead >= 1 h); one table each.
Table sets       = "sRMSE"  (per-pixel standardised RMSE, model + each baseline)
                   "RMSE"   (raw RMSE, model + each baseline)
                   "skill"  (1 - RMSE_model / RMSE_baseline, one table per baseline).

Aggregation: for every (init_time, pixel) we reduce the leads in the window to a
single RMSE, giving an empirical distribution over init x space. We report its
median and the 5th/95th percentiles as the CI. No bootstrap.

Standardisation (sRMSE set): divide each pixel's RMSE by that pixel's temporal
std of observations (over the window's init x lead samples). Per-pixel, NOT a
common std -- see results/verification/std_comparison_sim.py for why.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.evaluator import EvaluatorVerification
from deep_lst.evaluation.metrics import METRIC_REGISTRY

SPACE_DIMS = ("pixel", "lat", "lon")
NOWCAST_MAX_HOURS = 24
COLUMNS = ("validity", "daytime", "nighttime", "amplitude", "daily", "total")
QUANTILES = (0.05, 0.95)

# Region display names; "temporal" holdout reads the training store held out in time.
REGION_EVAL_NAME = {"temporal": "train_mtg"}
REGION_DISPLAY = {
    "temporal": "temporal",
    "antananarivo": "antananarivo",
    "johannesburg": "johannesburg",
    "yangambi": "yangambi",
    "bouamir": "bouamir",
    "dahra": "dahra",
    "gobabeb": "gobabeb",
    "sahel_niger": "sahel",
    "west_africa": "west africa",
    "passage": "passage",
}


# --- small shared helpers ----------------------------------------------------
def _space_dims(da: xr.DataArray) -> list[str]:
    return [d for d in SPACE_DIMS if d in da.dims]


def _lead_hours(da: xr.DataArray) -> np.ndarray:
    return (da["lead_time"].values / np.timedelta64(1, "h")).astype(int)


def _select_window(da: xr.DataArray, window: str) -> xr.DataArray:
    lead_hours = _lead_hours(da)
    if window == "nowcast":
        keep = (lead_hours >= 1) & (lead_hours <= NOWCAST_MAX_HOURS)
    else:
        keep = lead_hours >= 1
    return da.isel(lead_time=np.where(keep)[0])


def _valid_hour(da: xr.DataArray) -> xr.DataArray:
    """Hour-of-day of each (init_time, lead_time) valid time, as a 2D field."""
    init_hour = pd.DatetimeIndex(da["init_time"].values).hour
    init_hour = xr.DataArray(init_hour, dims="init_time", coords={"init_time": da["init_time"]})
    lead_hour = xr.DataArray(_lead_hours(da), dims="lead_time", coords={"lead_time": da["lead_time"]})
    return (init_hour + lead_hour) % 24


def _mask_time_of_day(da: xr.DataArray, kind: str) -> xr.DataArray:
    valid_hour = _valid_hour(da)
    if kind == "day":
        mask = (valid_hour >= 10) & (valid_hour <= 15)
    else:
        mask = (valid_hour >= 22) | (valid_hour <= 3)
    return da.where(mask)


def _daily_blocks(da: xr.DataArray) -> xr.DataArray:
    """Reshape consecutive leads into (day, hour) blocks of 24 h, dropping a partial tail.

    coarsen.construct is a vectorised reshape -- far faster than groupby-by-day,
    which dominated runtime. Assumes leads are consecutive hourly (true after
    _select_window, which keeps lead 1 h onward).
    """
    return da.coarsen(lead_time=24, boundary="trim").construct(lead_time=("day", "hour"))


# --- data transforms feeding the metric (per-pixel, reduce lead_time or day) -
def _daily_mean(da: xr.DataArray) -> xr.DataArray:
    return _daily_blocks(da).mean("hour")


def _diurnal_range(blocks: xr.DataArray) -> xr.DataArray:
    """q95 - q05 over the 'hour' axis, via one sort (np.nanquantile is ~400x slower).

    NaNs sort to the end, so the first n_valid entries are the valid hours; we read
    each quantile at the linearly-interpolated rank q*(n_valid-1) -- matching the
    default 'linear' method of np.nanquantile.
    """
    other_dims = [dim for dim in blocks.dims if dim != "hour"]
    values = np.sort(blocks.transpose(*other_dims, "hour").values, axis=-1)
    n_valid = np.isfinite(values).sum(axis=-1)

    def quantile_at(q):
        rank = q * np.maximum(n_valid - 1, 0)
        low_idx = np.floor(rank).astype(np.int64)
        high_idx = np.ceil(rank).astype(np.int64)
        frac = rank - low_idx
        low_val = np.take_along_axis(values, low_idx[..., None], axis=-1)[..., 0]
        high_val = np.take_along_axis(values, high_idx[..., None], axis=-1)[..., 0]
        return low_val + frac * (high_val - low_val)

    amplitude = np.where(n_valid > 0, quantile_at(0.95) - quantile_at(0.05), np.nan)
    coords = {dim: blocks[dim] for dim in other_dims if dim in blocks.coords}
    return xr.DataArray(amplitude, dims=other_dims, coords=coords)


def _daily_amplitude(da: xr.DataArray) -> xr.DataArray:
    """Robust diurnal range (q95 - q05) within each forecast-day block."""
    return _diurnal_range(_daily_blocks(da))


def column_samples(pred: xr.DataArray, obs: xr.DataArray, metric: str) -> dict[str, xr.DataArray]:
    """Per-(init, pixel) metric value for each distribution column.

    The metric (rmse, bias, mae, ...) comes from metrics.METRIC_REGISTRY, so the
    table is metric-agnostic. daytime/nighttime reduce lead_time under an hour
    mask; daily/amplitude first collapse leads into the daily mean / daily range,
    then reduce over the day blocks.
    """
    metric_fn = METRIC_REGISTRY[metric]
    return {
        "daytime": metric_fn(_mask_time_of_day(pred, "day"), _mask_time_of_day(obs, "day"), dim="lead_time"),
        "nighttime": metric_fn(_mask_time_of_day(pred, "night"), _mask_time_of_day(obs, "night"), dim="lead_time"),
        "amplitude": metric_fn(_daily_amplitude(pred), _daily_amplitude(obs), dim="day"),
        "daily": metric_fn(_daily_mean(pred), _daily_mean(obs), dim="day"),
    }


def total_value(pred: xr.DataArray, obs: xr.DataArray, metric: str) -> float:
    """Single metric value pooled over all init, lead and space samples."""
    return float(METRIC_REGISTRY[metric](pred, obs, dim=list(pred.dims)))


# --- standardisation and summary --------------------------------------------
def temporal_std(obs: xr.DataArray) -> xr.DataArray:
    """Per-pixel temporal std of obs over the window's init x lead samples."""
    return obs.std(dim=["init_time", "lead_time"], skipna=True)


def summarise(sample: xr.DataArray, std: xr.DataArray | None) -> tuple[float, float, float]:
    if std is not None:
        sample = sample / std
    reduce_dims = [d for d in ("init_time", *SPACE_DIMS) if d in sample.dims]
    median = float(sample.median(dim=reduce_dims))
    q_low = float(sample.quantile(QUANTILES[0], dim=reduce_dims))
    q_high = float(sample.quantile(QUANTILES[1], dim=reduce_dims))
    return median, q_low, q_high


def validity_pct(obs: xr.DataArray) -> float:
    finite_fraction = obs.notnull().mean(dim=["init_time", "lead_time"]) * 100.0
    return float(finite_fraction.median(dim=_space_dims(obs)))


# --- cell formatting ---------------------------------------------------------
def _fmt_ci(median: float, q_low: float, q_high: float) -> str:
    return f"{median:.3f} [{q_low:.3f}, {q_high:.3f}]"


# --- row builders (numeric core + formatting) --------------------------------
DISTRIBUTION_COLUMNS = ("daytime", "nighttime", "amplitude", "daily")


def row_values(pred: xr.DataArray, obs: xr.DataArray, metric: str, standardise: bool) -> dict:
    """Numeric row: {validity: float, <dist col>: (median, q05, q95), total: float}.

    standardise divides each pixel's metric by that pixel's temporal std(obs).
    """
    std = temporal_std(obs) if standardise else None
    samples = column_samples(pred, obs, metric)

    values = {"validity": validity_pct(obs)}
    for column, sample in samples.items():
        values[column] = summarise(sample, std)

    total = total_value(pred, obs, metric)
    if standardise:
        total = total / float(obs.std(skipna=True))
    values["total"] = total
    return values


def format_row(values: dict) -> dict[str, str]:
    row = {"validity": f"{values['validity']:.1f}\\%", "total": f"{values['total']:.3f}"}
    for column in DISTRIBUTION_COLUMNS:
        row[column] = _fmt_ci(*values[column])
    return row


def skill_row(model_values: dict, baseline_values: dict) -> dict[str, str]:
    """Skill 1 - metric_model / metric_baseline, from the per-column medians (point, no CI)."""
    row = {"validity": f"{model_values['validity']:.1f}\\%"}
    for column in DISTRIBUTION_COLUMNS:
        row[column] = f"{1.0 - model_values[column][0] / baseline_values[column][0]:.3f}"
    row["total"] = f"{1.0 - model_values['total'] / baseline_values['total']:.3f}"
    return row


# --- data loading ------------------------------------------------------------
def load_entities(experiment, version, eval_name, var, baselines, temporal):
    """Return (obs, {entity_name: prediction_da}), all materialised in memory.

    The forecast zarr is chunked with the whole lead axis in one chunk, so the
    evaluator's per-lead baseline masking would re-read every chunk once per lead
    (tens of minutes per region). We load the test-period forecast ONCE, then reuse
    the evaluator's own alignment (`_process_baseline`) against that in-memory array,
    which makes each baseline a few seconds and gives identical results.
    """
    evaluator = EvaluatorVerification(
        experiment=experiment,
        version=version,
        eval_name=eval_name,
        var=var,
        baseline=None,
        temporal=temporal,
        aggregation="region_time",
        consistent_mask=False,
        overwrite=False,
        metrics=("rmse",),
        compute_anomaly=False,
    )
    forecast_ds = evaluator._filter(evaluator._load_forecast()).load()
    evaluator.pred = forecast_ds[f"{var}_pred"]
    evaluator.obs = forecast_ds[f"{var}_obs"]

    obs = evaluator.obs
    predictions = {"model": evaluator.pred}
    lead_times = forecast_ds["lead_time"].values

    for baseline in baselines:
        evaluator.baseline = baseline
        evaluator.baseline_ds = evaluator._load_baseline()
        evaluator.baseline_ds = evaluator._subset_pixels(evaluator.baseline_ds, evaluator.pred.isel(lead_time=0))
        slices = [
            evaluator._process_baseline(evaluator.pred.sel(lead_time=lead_time), lead_time) for lead_time in lead_times
        ]
        baseline_full = xr.concat(slices, dim=pd.Index(lead_times, name="lead_time"))
        predictions[baseline] = baseline_full.load()
    return obs, predictions


# --- table assembly ----------------------------------------------------------
def to_latex(rows: dict[str, dict[str, str]], caption: str, label: str) -> str:
    col_spec = "c|" + "|".join("c" for _ in COLUMNS)
    header = " & " + " & ".join(COLUMNS) + r" \\"
    lines = [r"\begin{table}[]", r"    \centering", f"    \\begin{{tabular}}{{{col_spec}}}", "    " + header]
    for region_key, row in rows.items():
        display = REGION_DISPLAY.get(region_key, region_key)
        cells = " & ".join(row[column] for column in COLUMNS)
        lines.append(f"    {display} & {cells} \\\\")
    lines += [r"    \end{tabular}", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _caption(metric, kind, entity, window):
    if kind == "skill":
        return f"Skill (1 - {metric}/{metric}) of the model against {entity}, {window} window."
    if kind == "std":
        return f"Standardised {metric} (per-pixel / std(obs)) for {entity}, {window} window. Median [5th, 95th percentile] over init times and space."
    return f"{metric} for {entity}, {window} window. Median [5th, 95th percentile] over init times and space."


def build_tables(experiment, version, regions, var, baselines, temporal, windows, metrics, kinds):
    """Build every requested table, loading each region's data only once.

    Returns {(metric, kind, window, entity): {region_key: formatted_row}}.
    """
    tables_rows: dict[tuple, dict[str, dict[str, str]]] = {}

    for region_key in regions:
        eval_name = REGION_EVAL_NAME.get(region_key, region_key)
        try:
            obs_full, predictions = load_entities(experiment, version, eval_name, var, baselines, temporal)
        except FileNotFoundError as error:
            print(f"  [skip] {region_key}: {error}")
            continue

        for window in windows:
            # Load the windowed slice into memory: percentiles over multiple dims
            # are not supported on dask arrays, and test periods are small.
            obs = _select_window(obs_full, window).load()
            windowed = {name: _select_window(da, window).load() for name, da in predictions.items()}

            for metric in metrics:
                raw_values = {name: row_values(da, obs, metric, standardise=False) for name, da in windowed.items()}
                std_values = (
                    {name: row_values(da, obs, metric, standardise=True) for name, da in windowed.items()}
                    if "std" in kinds
                    else {}
                )
                for kind in kinds:
                    if kind == "raw":
                        for entity, values in raw_values.items():
                            tables_rows.setdefault((metric, kind, window, entity), {})[region_key] = format_row(values)
                    elif kind == "std":
                        for entity, values in std_values.items():
                            tables_rows.setdefault((metric, kind, window, entity), {})[region_key] = format_row(values)
                    elif kind == "skill":
                        for baseline in baselines:
                            row = skill_row(raw_values["model"], raw_values[baseline])
                            tables_rows.setdefault((metric, kind, window, baseline), {})[region_key] = row
    return tables_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--variable", default="LST")
    parser.add_argument("--regions", nargs="+", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--temporal", nargs="+", default=["test_period_00"])
    parser.add_argument("--windows", nargs="+", default=["nowcast", "full"])
    parser.add_argument("--metrics", nargs="+", default=["rmse"], help="any name in metrics.METRIC_REGISTRY")
    parser.add_argument("--sets", nargs="+", default=["raw", "std", "skill"], choices=["raw", "std", "skill"])
    args = parser.parse_args()

    out_dir = Path("results") / "tables" / f"{args.experiment}_{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_rows = build_tables(
        args.experiment,
        args.version,
        args.regions,
        args.variable,
        args.baselines,
        args.temporal,
        args.windows,
        args.metrics,
        args.sets,
    )

    for (metric, kind, window, entity), rows in tables_rows.items():
        name = f"{metric}_{kind}_{entity}_{window}"
        latex = to_latex(rows, _caption(metric, kind, entity, window), f"tab:{name}")
        print(f"\n===== {name} =====\n{latex}")
        out_path = out_dir / f"{name}.tex"
        out_path.write_text(latex + "\n")
        print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
