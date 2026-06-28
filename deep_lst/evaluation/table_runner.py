"""Score tables across regions and temporal scales, in LaTeX-ready form.

Design goal: flat and readable, so every cell traces to one small function.

Rows of a table  = holdout regions (plus the "temporal" holdout).
Columns          = temporal scales: validity | hourly | daytime | nighttime
                   | amplitude | daily | total.
Lead windows     = "nowcast" (lead 1-24 h), "full" (lead >= 1 h), and "dayN" (the
                   24 h of forecast day N, e.g. day1 = 1-24 h, day10 = 217-240 h);
                   one table each.
Table sets       = "sRMSE"  (standardised RMSE, model + each baseline)
                   "RMSE"   (raw RMSE, model + each baseline)
                   "skill"  (1 - RMSE_model / RMSE_baseline, one table per baseline).

Aggregation: every cell is a single metric value pooled over all samples in the
window (init x lead x space) in the squared-error domain -- the climpred /
xskillscore philosophy. No per-(init, pixel) distribution, no CIs.

Standardisation (sRMSE set): divide each column's pooled metric by the std of
observations pooled over the SAME transformed samples, so numerator and
denominator collapse the same dimensions. This supersedes the earlier per-pixel
std (see results/verification/std_comparison_sim.py for that prior rationale).
"""

import argparse
import json

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.evaluator import EvaluatorVerification
from deep_lst.evaluation.latex import latex_table, write_month_tables
from deep_lst.evaluation.metrics import pool_metric, to_lead_hours
from deep_lst.evaluation.plots.month_grid import MONTHS, _latitude_labels, continuous_norm, month_grid_heatmap
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

NOWCAST_MAX_HOURS = 24
COLUMNS = ("validity", "daytime", "nighttime", "amplitude", "daily", "total")

# Monthly-RMSE mode (--monthly): same pooling as the score tables, split by init month.
DAYNIGHT = ("day", "night")
TRAINING_REGION_KEY = "temporal"  # the train_mtg holdout column in REGION_DISPLAY

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
def _day_window_bounds(window: str) -> tuple[int, int] | None:
    """Inclusive lead-hour bounds for a 'dayN' window (24 h of forecast day N), else None."""
    if not window.startswith("day") or not window[3:].isdigit():
        return None
    day = int(window[3:])
    return (day - 1) * 24 + 1, day * 24


def _select_window(da: xr.DataArray, window: str) -> xr.DataArray:
    lead_hours = to_lead_hours(da)
    day_bounds = _day_window_bounds(window)
    if window == "nowcast":
        keep = (lead_hours >= 1) & (lead_hours <= NOWCAST_MAX_HOURS)
    elif day_bounds is not None:
        low, high = day_bounds
        keep = (lead_hours >= low) & (lead_hours <= high)
    else:
        keep = lead_hours >= 1
    return da.isel(lead_time=np.where(keep)[0])


def _valid_hour(da: xr.DataArray) -> xr.DataArray:
    """Hour-of-day of each (init_time, lead_time) valid time, as a 2D field."""
    init_hour = pd.DatetimeIndex(da["init_time"].values).hour
    init_hour = xr.DataArray(init_hour, dims="init_time", coords={"init_time": da["init_time"]})
    lead_hour = xr.DataArray(to_lead_hours(da), dims="lead_time", coords={"lead_time": da["lead_time"]})
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


def _column_fields(pred: xr.DataArray, obs: xr.DataArray, column: str):
    """Transform pred/obs for one metric column before the pooled reduction.

    daytime/nighttime mask leads to a valid-hour window (still hourly); daily takes
    the daily-mean field (RMSE of daily means); amplitude takes the daily diurnal
    range; total keeps the raw hourly fields.

    For daily/amplitude the daily means are formed AFTER masking pred to the obs-valid
    hours, so pred and obs are averaged over the same (clear-sky) hours. Otherwise pred's
    daily mean spans all 24 h while obs's spans only clear-sky hours, and LST's strong,
    diurnally-biased cloud sampling injects a spurious daily error (daily > total).
    """
    if column == "daytime":
        return _mask_time_of_day(pred, "day"), _mask_time_of_day(obs, "day")
    if column == "nighttime":
        return _mask_time_of_day(pred, "night"), _mask_time_of_day(obs, "night")
    if column == "amplitude":
        pred = pred.where(obs.notnull())
        return _daily_amplitude(pred), _daily_amplitude(obs)
    if column == "daily":
        pred = pred.where(obs.notnull())
        return _daily_mean(pred), _daily_mean(obs)
    return pred, obs


def pooled_metric(pred: xr.DataArray, obs: xr.DataArray, metric: str) -> float:
    """Single metric value pooled over every dim (squared-error domain)."""
    return float(pool_metric(pred, obs, metric, list(pred.dims)))


def pooled_std(obs: xr.DataArray) -> float:
    """Std of obs pooled over every dim -- the denominator matched to pooled_metric."""
    return float(obs.std(skipna=True))


def validity_pct(obs: xr.DataArray) -> float:
    """Percentage of valid obs, pooled over every dim."""
    return float(obs.notnull().mean()) * 100.0


# --- row builders (numeric core + formatting) --------------------------------
METRIC_COLUMNS = ("daytime", "nighttime", "amplitude", "daily", "total")


def row_values(pred: xr.DataArray, obs: xr.DataArray, metric: str, standardise: bool) -> dict:
    """Numeric row: {validity: float, <metric col>: float}; one pooled value per cell.

    standardise divides each column's pooled metric by the pooled std of obs over the
    same transformed samples (matched aggregation).
    """
    values = {"validity": validity_pct(obs)}
    for column in METRIC_COLUMNS:
        pred_column, obs_column = _column_fields(pred, obs, column)
        value = pooled_metric(pred_column, obs_column, metric)
        if standardise:
            value = value / pooled_std(obs_column)
        values[column] = value
    return values


def format_row(values: dict) -> dict[str, str]:
    row = {"validity": f"{values['validity']:.1f}\\%"}
    for column in METRIC_COLUMNS:
        row[column] = f"{values[column]:.3f}"
    return row


def skill_row(model_values: dict, baseline_values: dict) -> dict[str, str]:
    """Skill 1 - metric_model / metric_baseline, per pooled column value."""
    row = {"validity": f"{model_values['validity']:.1f}\\%"}
    for column in METRIC_COLUMNS:
        row[column] = f"{1.0 - model_values[column] / baseline_values[column]:.3f}"
    return row


# --- data loading ------------------------------------------------------------
def load_entities(experiment, version, eval_name, var, baselines, temporal, attach_classes=False):
    """Return (obs, {entity_name: prediction_da}), all materialised in memory.

    The forecast zarr is chunked with the whole lead axis in one chunk, so the
    evaluator's per-lead baseline masking would re-read every chunk once per lead
    (tens of minutes per region). We load the test-period forecast ONCE, then reuse
    the evaluator's own alignment (`_baseline_full`) against that in-memory array,
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
    forecast_ds = evaluator._filter(evaluator._load_forecast())
    if attach_classes:
        forecast_ds = evaluator._attach_class_coords(forecast_ds)  # per-pixel class coords (kg, modis_lc, ...)
    forecast_ds = forecast_ds.load()
    evaluator.pred = forecast_ds[f"{var}_pred"]
    evaluator.obs = forecast_ds[f"{var}_obs"]

    obs = evaluator.obs
    predictions = {"model": evaluator.pred}

    for baseline in baselines:
        evaluator.baseline = baseline
        evaluator.baseline_ds = evaluator._load_baseline()
        evaluator.baseline_ds = evaluator._subset_pixels(evaluator.baseline_ds, evaluator.pred.isel(lead_time=0))
        baseline_full = evaluator._baseline_full()
        predictions[baseline] = baseline_full.load()
    return obs, predictions


# --- table assembly ----------------------------------------------------------
def _caption(metric, kind, entity, window):
    if kind == "skill":
        return f"Skill (1 - {metric}/{metric}) of the model against {entity}, {window} window."
    if kind == "std":
        return f"Standardised {metric} ({metric} / std(obs), pooled over init times, leads and space) for {entity}, {window} window."
    return f"{metric} pooled over init times, leads and space for {entity}, {window} window."


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
            if obs.sizes.get("lead_time", 0) == 0:
                print(f"  [skip] {region_key} {window}: no lead times in window")
                continue
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


# --- monthly-RMSE mode: pooled RMSE per (region|class, month, day/night, window) ------------------
def _month_window_rmse(model, obs, month, kind, window):
    """Pooled RMSE [K] for one (init-month, day/night, lead-window) cell, over init x leads x space.

    Reuses the score-table pooling (`_select_window`, `_mask_time_of_day`, `pooled_metric`); the only
    extra axis is the init-month groupby. Returns NaN when the masked window has no valid obs.
    """
    init_months = pd.DatetimeIndex(model["init_time"].values).month
    in_month = init_months == month
    model_sel = _select_window(model.isel(init_time=in_month), window)
    obs_sel = _select_window(_mask_time_of_day(obs.isel(init_time=in_month), kind), window)
    if int(obs_sel.notnull().sum()) == 0:
        return float("nan")
    return pooled_metric(model_sel, obs_sel, "rmse")


def monthly_rmse_records(experiment, version, region_specs, temporal, windows, var="LST"):
    """One record per (region, month, day/night, window): pooled RMSE. region_specs: (label, eval_name)."""
    records = []
    for label, eval_name in region_specs:
        try:
            obs, predictions = load_entities(experiment, version, eval_name, var, [], temporal)
        except FileNotFoundError as error:
            print(f"  [skip] {label} ({eval_name}): {error}")
            continue
        model = predictions["model"]
        latitude = float(obs["lat"].mean())
        for month in MONTHS:
            for kind in DAYNIGHT:
                for window in windows:
                    records.append(
                        {
                            "region": label,
                            "latitude": latitude,
                            "month": month,
                            "daynight": kind,
                            "window": window,
                            "rmse": _month_window_rmse(model, obs, month, kind, window),
                        }
                    )
        print(f"  {label:<14} done")
    return records


def monthly_rmse_class_records(
    experiment, version, eval_name, class_key, class_labels, temporal, windows, var="LST", min_pixels=20
):
    """One record per (class, month, day/night, window) for one site, pooling pixels by `class_key`."""
    obs, predictions = load_entities(experiment, version, eval_name, var, [], temporal, attach_classes=True)
    model = predictions["model"]
    class_coord = model[class_key]

    records = []
    for class_id in np.unique(class_coord.values):
        if class_id == -1:
            continue
        pixel_mask = (class_coord == class_id).values
        if pixel_mask.sum() < min_pixels:
            print(
                f"  [skip] class {int(class_id):>3}  {class_labels.get(str(int(class_id)), class_id)} ({int(pixel_mask.sum())} px < {min_pixels})"
            )
            continue
        model_class = model.isel(pixel=pixel_mask)
        obs_class = obs.isel(pixel=pixel_mask)
        label = class_labels.get(str(int(class_id)), str(int(class_id)))
        for month in MONTHS:
            for kind in DAYNIGHT:
                for window in windows:
                    records.append(
                        {
                            "region": label,
                            "class_id": int(class_id),
                            "month": month,
                            "daynight": kind,
                            "window": window,
                            "rmse": _month_window_rmse(model_class, obs_class, month, kind, window),
                        }
                    )
        print(f"  class {int(class_id):>3}  {label:<32} ({int(pixel_mask.sum())} px) done")
    return records


def _rmse_cell(record):
    """LaTeX table cell: RMSE to two decimals, '--' when empty."""
    value = record["rmse"]
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.2f}"


def run_monthly(args):
    """Monthly pooled-RMSE heatmaps + tables for regions and land-cover classes (shared scale/window)."""
    region_specs = [(REGION_DISPLAY[key], REGION_EVAL_NAME.get(key, key)) for key in REGION_DISPLAY]
    training_label = REGION_DISPLAY[TRAINING_REGION_KEY]
    windows = tuple(args.windows)

    region_df = pd.DataFrame.from_records(
        monthly_rmse_records(args.experiment, args.version, region_specs, args.temporal, windows, args.variable)
    )
    class_labels = json.loads(PATHS.eval_meta_path("seviri").read_text())[f"{args.by_class}_labels"]
    class_df = pd.DataFrame.from_records(
        monthly_rmse_class_records(
            args.experiment,
            args.version,
            args.class_eval_name,
            args.by_class,
            class_labels,
            args.temporal,
            windows,
            args.variable,
        )
    )

    tables_dir = PATHS.tables_dir(args.experiment, args.version, "monthly_rmse")
    figures_dir = PATHS.figures_dir(args.experiment, args.version)

    # region column order: training holdout first, then eval regions by descending latitude.
    lat_by_region = region_df.groupby("region")["latitude"].first()
    eval_regions = [region for region in lat_by_region.index if region != training_label]
    region_order = [training_label] + sorted(eval_regions, key=lat_by_region.get, reverse=True)
    region_top = _latitude_labels(region_df, region_order, training_label)
    class_order = class_df[["region", "class_id"]].drop_duplicates().sort_values("class_id")["region"].tolist()

    for window in windows:
        region_win = region_df[region_df["window"] == window]
        class_win = class_df[class_df["window"] == window]
        # one colour scale per window, shared across the region and class heatmaps.
        both = np.concatenate([region_win["rmse"].to_numpy(), class_win["rmse"].to_numpy()])
        cmap, norm = continuous_norm(float(np.nanmin(both)), float(np.nanmax(both)))
        stem = f"{args.experiment}_{args.version}_{window}"

        region_win.to_csv(tables_dir / f"{stem}_regions.csv", index=False)
        class_win.to_csv(tables_dir / f"{stem}_{args.by_class}.csv", index=False)
        write_month_tables(
            region_win.to_dict("records"),
            region_order,
            tables_dir,
            f"{stem}_regions",
            f"Monthly RMSE [K] ({window}) by region for {args.experiment}",
            f"tab:{stem}_regions",
            _rmse_cell,
            MONTHS,
            DAYNIGHT,
        )
        write_month_tables(
            class_win.to_dict("records"),
            class_order,
            tables_dir,
            f"{stem}_{args.by_class}",
            f"Monthly RMSE [K] ({window}) by {args.by_class} for {args.experiment}",
            f"tab:{stem}_{args.by_class}",
            _rmse_cell,
            MONTHS,
            DAYNIGHT,
        )
        month_grid_heatmap(
            region_win,
            region_order,
            "rmse",
            figures_dir / f"{stem}_regions.pdf",
            cmap=cmap,
            norm=norm,
            cbar_label="RMSE [K]",
            top_labels=region_top,
            separator_x=0.5,
            col_gridlines=True,
            figsize_per_col=1.2,
            figsize_base=2.0,
        )
        month_grid_heatmap(
            class_win,
            class_order,
            "rmse",
            figures_dir / f"{stem}_{args.by_class}.pdf",
            cmap=cmap,
            norm=norm,
            cbar_label="RMSE [K]",
            figsize_per_col=0.55,
            figsize_base=3.0,
        )
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--variable", default="LST")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--baselines", nargs="+", default=None)
    parser.add_argument("--temporal", nargs="+", default=["test_period_00"])
    parser.add_argument("--windows", nargs="+", default=["nowcast", "full"])
    parser.add_argument("--metrics", nargs="+", default=["rmse"], help="any name in metrics.METRIC_REGISTRY")
    parser.add_argument("--sets", nargs="+", default=["raw", "std", "skill"], choices=["raw", "std", "skill"])
    parser.add_argument(
        "--monthly",
        action="store_true",
        help="Monthly pooled-RMSE heatmaps + tables (regions + classes) instead of the score tables.",
    )
    parser.add_argument("--by-class", default="modis_lc_classes", help="Class coord pooled in --monthly class heatmap.")
    parser.add_argument(
        "--class-eval-name", default="train_mtg", help="Site whose pixels are pooled in --monthly class mode."
    )
    args = parser.parse_args()

    if args.monthly:
        run_monthly(args)
        return

    if args.regions is None or args.baselines is None:
        parser.error("--regions and --baselines are required unless --monthly is set")

    out_dir = PATHS.tables_dir(args.experiment, args.version, "scores")

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
        latex = latex_table(rows, COLUMNS, _caption(metric, kind, entity, window), f"tab:{name}", REGION_DISPLAY)
        print(f"\n===== {name} =====\n{latex}")
        out_path = out_dir / f"{name}.tex"
        out_path.write_text(latex + "\n")
        print(f"[written] {out_path}")


if __name__ == "__main__":
    main()
