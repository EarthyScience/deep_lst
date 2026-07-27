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

from deep_lst.evaluation.conditioning import mask_time_of_day
from deep_lst.evaluation.evaluator import EvaluatorVerification
from deep_lst.evaluation.latex import latex_table, write_month_tables
from deep_lst.evaluation.metrics import pool_metric, skill_score, to_lead_hours
from deep_lst.evaluation.plots.month_grid import (
    MONTHS,
    _latitude_labels,
    fingerprint_decomposition_heatmap,
    fingerprint_decomposition_honest_heatmap,
    fingerprint_rmse_heatmap,
    monthly_rmse_heatmap,
)
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
        return mask_time_of_day(pred, "day"), mask_time_of_day(obs, "day")
    if column == "nighttime":
        return mask_time_of_day(pred, "night"), mask_time_of_day(obs, "night")
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
        row[column] = f"{skill_score(model_values[column], baseline_values[column]):.3f}"
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
    )
    evaluator.load()  # class coords are always attached now; attach_classes kept for API compatibility
    evaluator.pred = evaluator.pred.load()
    evaluator.obs = evaluator.obs.load()

    predictions = {"model": evaluator.pred}
    for baseline in baselines:
        predictions[baseline] = evaluator.load_baseline(baseline).load()
    return evaluator.obs, predictions


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

    Reuses the score-table pooling (`_select_window`, `mask_time_of_day`, `pooled_metric`); the only
    extra axis is the init-month groupby. Returns NaN when the masked window has no valid obs.
    """
    init_months = pd.DatetimeIndex(model["init_time"].values).month
    in_month = init_months == month
    model_sel = _select_window(model.isel(init_time=in_month), window)
    obs_sel = _select_window(mask_time_of_day(obs.isel(init_time=in_month), kind), window)
    if int(obs_sel.notnull().sum()) == 0:
        return float("nan")
    return pooled_metric(model_sel, obs_sel, "rmse")


def _month_window_rmse_decomp(model, obs, month, kind, window):
    """(raw, debiased) pooled RMSE [K] for one (init-month, day/night, lead-window) cell.

    debiased removes each pixel's mean error within the cell (per-pixel centering, _pooled_rmse_decomp).
    These cells pool the whole window x all init-days x space, so per-pixel n is large and the in-sample
    DoF artifact is negligible -- the cleanest place to report a debiased RMSE.
    """
    init_months = pd.DatetimeIndex(model["init_time"].values).month
    in_month = init_months == month
    model_sel = _select_window(model.isel(init_time=in_month), window)
    obs_sel = _select_window(mask_time_of_day(obs.isel(init_time=in_month), kind), window)
    raw, debiased, _ = _pooled_rmse_decomp(model_sel, obs_sel)
    return raw, debiased


def _bias_rmse(raw, debiased):
    """Removed-bias RMSE [K] = sqrt(raw^2 - debiased^2) from the raw^2 = debiased^2 + bias^2 split."""
    if raw is None or debiased is None or not (np.isfinite(raw) and np.isfinite(debiased)):
        return float("nan")
    return float(np.sqrt(max(raw**2 - debiased**2, 0.0)))


def monthly_rmse_records(experiment, version, region_specs, temporal, windows, var="LST"):
    """One record per (region, month, day/night, window): raw + per-pixel-debiased pooled RMSE."""
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
                    raw, debiased = _month_window_rmse_decomp(model, obs, month, kind, window)
                    records.append(
                        {
                            "region": label,
                            "latitude": latitude,
                            "month": month,
                            "daynight": kind,
                            "window": window,
                            "rmse": raw,
                            "rmse_debiased": debiased,
                            "rmse_bias": _bias_rmse(raw, debiased),
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


def _rmse_debiased_cell(record):
    """LaTeX table cell: per-pixel-debiased RMSE to two decimals, '--' when empty."""
    value = record["rmse_debiased"]
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
            region_win.to_dict("records"),
            region_order,
            tables_dir,
            f"{stem}_regions_debiased",
            f"Monthly per-pixel-debiased RMSE [K] ({window}) by region for {args.experiment}",
            f"tab:{stem}_regions_debiased",
            _rmse_debiased_cell,
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
        # raw + debiased share ONE K scale (debiased then visibly <= raw); bias gets its own scale.
        both = np.concatenate([region_win["rmse"].to_numpy(float), region_win["rmse_debiased"].to_numpy(float)])
        shared = (float(np.nanmin(both)), float(np.nanmax(both)))
        monthly_rmse_heatmap(
            region_win,
            region_order,
            figures_dir / f"{stem}_regions.pdf",
            value_col="rmse",
            value_range=shared,
            top_labels=region_top,
            separator_x=0.5,
            col_gridlines=True,
            figsize_per_col=1.2,
            figsize_base=2.0,
        )
        monthly_rmse_heatmap(
            region_win,
            region_order,
            figures_dir / f"{stem}_regions_debiased.pdf",
            value_col="rmse_debiased",
            value_range=shared,
            cbar_label="RMSE debiased [K]",
            top_labels=region_top,
            separator_x=0.5,
            col_gridlines=True,
            figsize_per_col=1.2,
            figsize_base=2.0,
        )
        monthly_rmse_heatmap(
            region_win,
            region_order,
            figures_dir / f"{stem}_regions_bias.pdf",
            value_col="rmse_bias",
            cbar_label="Bias RMSE [K]",
            top_labels=region_top,
            separator_x=0.5,
            col_gridlines=True,
            figsize_per_col=1.2,
            figsize_base=2.0,
        )
        monthly_rmse_heatmap(
            class_win,
            class_order,
            figures_dir / f"{stem}_{args.by_class}.pdf",
            figsize_per_col=0.55,
            figsize_base=3.0,
        )
    print("\nDone.")


# --- fingerprint mode: month x hour-of-day RMSE per region (one panel per lead window) ------------
def _month_hour_suffstats(model, obs, window):
    """Per-pixel-per-cell sufficient stats on the month x hour grid, for one lead window.

    Returns sum_e (S1=Sigma e), sum_e2 (S2=Sigma e^2), count (n) each (288, n_pixel) with
    cell = (month-1)*24 + hour, plus per-cell n_samples (forecast (init, lead) pairs in the cell) and
    n_distinct (unique target timestamps) for the pseudo-replication check. error e = pred - obs.
    Shared by the raw/centered decomposition and the Bessel-corrected one (no model re-evaluation).
    """
    error = _select_window(model - obs, window)
    space = [dim for dim in error.dims if dim not in ("init_time", "lead_time")]
    values = error.transpose("init_time", "lead_time", *space).values
    n_pixel = int(np.prod(values.shape[2:]))
    flat_error = values.reshape(-1, n_pixel)  # (init*lead, pixel); NaN where obs missing

    init = pd.DatetimeIndex(error["init_time"].values).values
    lead = pd.TimedeltaIndex(error["lead_time"].values).values
    valid = pd.DatetimeIndex((init[:, None] + lead[None, :]).ravel())
    bucket = (valid.month.values - 1) * 24 + valid.hour.values  # 0..287, per (init, lead) cell
    valid_i8 = valid.asi8  # int64 ns, for counting distinct target timestamps

    finite = np.isfinite(flat_error)
    filled = np.where(finite, flat_error, 0.0)
    sum_e = np.zeros((288, n_pixel))
    sum_e2 = np.zeros((288, n_pixel))
    count = np.zeros((288, n_pixel))
    n_samples = np.zeros(288)
    n_distinct = np.zeros(288)
    for cell in range(288):
        rows = bucket == cell
        if not rows.any():
            continue
        sum_e[cell] = filled[rows].sum(axis=0)
        sum_e2[cell] = (filled[rows] ** 2).sum(axis=0)
        count[cell] = finite[rows].sum(axis=0)
        n_samples[cell] = int(rows.sum())
        n_distinct[cell] = np.unique(valid_i8[rows]).size
    return sum_e, sum_e2, count, n_samples, n_distinct


def _decomp_from_suffstats(sum_e, sum_e2, count):
    """(raw, centered) (24 hour x 12 month) RMSE [K] from suff stats; in-sample centering, as before."""
    total = count.sum(axis=1)
    raw_sse = sum_e2.sum(axis=1)
    per_pixel_ss = np.maximum(sum_e2 - np.divide(sum_e**2, count, out=np.zeros_like(sum_e), where=count > 0), 0.0)
    centered_sse = per_pixel_ss.sum(axis=1)
    raw = np.sqrt(np.divide(raw_sse, total, out=np.full(288, np.nan), where=total > 0)).reshape(12, 24)
    centered = np.sqrt(np.divide(centered_sse, total, out=np.full(288, np.nan), where=total > 0)).reshape(12, 24)
    return raw.T, centered.T  # each (hour, month)


def _month_hour_rmse_decomp(model, obs, window):
    """(raw, centered) RMSE [K] (24 hour x 12 month) for one window; in-sample centering (unchanged)."""
    sum_e, sum_e2, count, _, _ = _month_hour_suffstats(model, obs, window)
    return _decomp_from_suffstats(sum_e, sum_e2, count)


def _month_hour_honest(sum_e, sum_e2, count, n_samples, n_distinct):
    """Bessel-corrected decomposition + sampling diagnostics, all (24 hour, 12 month) grids.

    Drops pixels with n_p < 2 (SSdev undefined); Bessel weight n_p/(n_p-1) per pixel; raw is restricted
    to the SAME n_p >= 2 pixels (apples-to-apples). honest_collapse_pct and corrected_bias2 are returned
    UNCLAMPED -- <= 0 means the bias is statistically indistinguishable from zero.
    """
    used = count >= 2
    ssdev = np.maximum(sum_e2 - np.divide(sum_e**2, count, out=np.zeros_like(sum_e), where=count > 0), 0.0)

    n_used_sum = np.where(used, count, 0.0).sum(axis=1)  # Sigma n_p over n_p >= 2 pixels
    raw_sse = np.where(used, sum_e2, 0.0).sum(axis=1)
    bessel_num = np.where(used, count / np.maximum(count - 1.0, 1.0) * ssdev, 0.0).sum(axis=1)

    raw_mse = np.divide(raw_sse, n_used_sum, out=np.full(288, np.nan), where=n_used_sum > 0)
    bessel_mse = np.divide(bessel_num, n_used_sum, out=np.full(288, np.nan), where=n_used_sum > 0)
    corrected_bias2 = raw_mse - bessel_mse  # may be negative -- kept as-is
    honest_collapse_pct = 100.0 * np.divide(corrected_bias2, raw_mse, out=np.full(288, np.nan), where=raw_mse > 0)

    observed = count >= 1
    n_obs = observed.sum(axis=1).astype(float)
    dropped_frac = np.divide((count == 1).sum(axis=1).astype(float), n_obs, out=np.full(288, np.nan), where=n_obs > 0)
    median_n = np.nanmedian(np.where(observed, count, np.nan), axis=1)
    ratio = np.divide(n_samples, n_distinct, out=np.full(288, np.nan), where=n_distinct > 0)

    def grid(arr):
        return arr.reshape(12, 24).T

    return {
        "median_n": grid(median_n),
        "dropped_frac": grid(dropped_frac),
        "honest_collapse_pct": grid(honest_collapse_pct),
        "corrected_rmse": grid(np.sqrt(np.maximum(bessel_mse, 0.0))),
        "corrected_bias2": grid(corrected_bias2),
        "raw_restricted_rmse": grid(np.sqrt(np.maximum(raw_mse, 0.0))),
        "n_samples": grid(n_samples.astype(float)),
        "n_distinct": grid(n_distinct.astype(float)),
        "ratio": grid(ratio),
    }


def run_fingerprint(args):
    """Per-region month x hour-of-day RMSE fingerprints (one panel per lead window)."""
    windows = tuple(args.windows)
    tables_dir = PATHS.tables_dir(args.experiment, args.version, "fingerprint")
    figures_dir = PATHS.figures_dir(args.experiment, args.version) / "fingerprints"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment}_{args.version}"

    for key in REGION_DISPLAY:
        eval_name = REGION_EVAL_NAME.get(key, key)
        display = REGION_DISPLAY[key]
        try:
            obs, predictions = load_entities(args.experiment, args.version, eval_name, args.variable, [], args.temporal)
        except FileNotFoundError as error:
            print(f"  [skip] {display} ({eval_name}): {error}")
            continue
        model = predictions["model"]

        records, panels, decomp, honest_by_window = [], [], {}, {}
        for window in windows:
            sum_e, sum_e2, count, n_samples, n_distinct = _month_hour_suffstats(model, obs, window)
            raw, centered = _decomp_from_suffstats(sum_e, sum_e2, count)  # each (24 hour, 12 month)
            collapse = 100.0 * (1.0 - np.divide(centered**2, raw**2, out=np.full_like(raw, np.nan), where=raw > 0))
            honest = _month_hour_honest(sum_e, sum_e2, count, n_samples, n_distinct)
            decomp[window] = (raw, centered, collapse)
            honest_by_window[window] = honest
            panels.append((f"{display} — {window}", raw))
            for hour in range(24):
                for month in range(12):
                    records.append(
                        {
                            "region": display,
                            "eval_name": eval_name,
                            "window": window,
                            "month": month + 1,
                            "hour": hour,
                            "rmse": float(raw[hour, month]),
                            "rmse_centered": float(centered[hour, month]),
                            "collapse_pct": float(collapse[hour, month]),
                            "median_n": float(honest["median_n"][hour, month]),
                            "dropped_frac": float(honest["dropped_frac"][hour, month]),
                            "rmse_centered_bessel": float(honest["corrected_rmse"][hour, month]),
                            "honest_collapse_pct": float(honest["honest_collapse_pct"][hour, month]),
                            "corrected_bias2": float(honest["corrected_bias2"][hour, month]),
                            "raw_restricted_rmse": float(honest["raw_restricted_rmse"][hour, month]),
                            "n_samples": float(honest["n_samples"][hour, month]),
                            "n_distinct_targets": float(honest["n_distinct"][hour, month]),
                            "replication_ratio": float(honest["ratio"][hour, month]),
                        }
                    )
        pd.DataFrame.from_records(records).to_csv(tables_dir / f"{stem}_{key}.csv", index=False)
        fingerprint_rmse_heatmap(panels, figures_dir / f"{stem}_{key}_fingerprint.pdf")

        decomp_window = "full" if "full" in decomp else windows[-1]
        raw, centered, collapse = decomp[decomp_window]
        fingerprint_decomposition_heatmap(
            raw,
            centered,
            collapse,
            f"{display} — {decomp_window}: per-pixel bias decomposition",
            figures_dir / f"{stem}_{key}_decomposition.pdf",
        )
        honest = honest_by_window[decomp_window]
        fingerprint_decomposition_honest_heatmap(
            raw,
            centered,
            collapse,
            honest["median_n"],
            honest["honest_collapse_pct"],
            honest["corrected_rmse"],
            f"{display} — {decomp_window}: Bessel-corrected decomposition",
            figures_dir / f"{stem}_{key}_decomposition_honest.pdf",
        )
        _print_honest_summary(display, key, decomp[decomp_window], honest)
        print(f"  {display:<14} done")
    print("\nDone.")


def _print_honest_summary(display, key, decomp_full, honest):
    """For yangambi/sahel, print the peak-RMSE daytime cell: raw vs honest %collapsed, median n_p, ratio."""
    if key not in ("yangambi", "sahel_niger"):
        return
    raw, _, collapse = decomp_full
    daytime = np.full_like(raw, np.nan)
    daytime[10:16, :] = raw[10:16, :]  # hours 10-15 UTC — the fingerprint hour axis is UTC, not local solar
    if not np.isfinite(daytime).any():
        return
    hour, month = np.unravel_index(np.nanargmax(daytime), daytime.shape)
    print(
        f"  [summary] {display:<10} peak-day cell month={month + 1:02d} hour={hour:02d}: "
        f"rawRMSE={raw[hour, month]:.2f}K  raw%collapse={collapse[hour, month]:.0f}  "
        f"honest%collapse={honest['honest_collapse_pct'][hour, month]:.0f}  "
        f"median_n={honest['median_n'][hour, month]:.0f}  replication_ratio={honest['ratio'][hour, month]:.1f}"
    )


# --- decomposition tables: pooled raw RMSE | per-pixel-debiased RMSE | % MSE collapsed -------------
DECOMP_TODS = ("total", "daytime", "nighttime")


def _pooled_rmse_decomp(pred, obs):
    """(raw, debiased, collapse%) pooled RMSE; debiased removes each pixel's mean error first.

    Per-pixel bias = mean error over the column's (init, lead) samples for that pixel; debiased RMSE
    is sqrt of the per-pixel temporal variance pooled over pixels. raw^2 = debiased^2 + spatial-bias^2.
    """
    error = pred - obs
    reduce = [dim for dim in ("init_time", "lead_time") if dim in error.dims]
    count = error.notnull().sum(reduce)
    sum_e = error.sum(reduce, skipna=True)
    sum_e2 = (error**2).sum(reduce, skipna=True)
    total = float(count.sum())
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    raw = float(np.sqrt(float(sum_e2.sum()) / total))
    per_pixel_ss = (sum_e2 - sum_e**2 / count.where(count > 0)).clip(min=0)  # guard float cancellation
    debiased = float(np.sqrt(float(per_pixel_ss.sum()) / total))
    collapse = 100.0 * (1.0 - (debiased / raw) ** 2) if raw > 0 else float("nan")
    return raw, debiased, collapse


def _decomp_cells(model_window, obs_window):
    """Formatted {col: cell} for one (region, window): raw/debiased RMSE + collapse per time-of-day."""
    cells = {}
    for tod in DECOMP_TODS:
        pred_col, obs_col = _column_fields(model_window, obs_window, tod)
        raw, debiased, collapse = _pooled_rmse_decomp(pred_col, obs_col)
        cells[f"{tod} RMSE"] = "--" if np.isnan(raw) else f"{raw:.2f}"
        cells[f"{tod} deb"] = "--" if np.isnan(debiased) else f"{debiased:.2f}"
        cells[f"{tod} coll"] = "--" if np.isnan(collapse) else f"{collapse:.0f}\\%"
    return cells


def run_decomposition(args):
    """Compact per-region bias-decomposition tables (raw | debiased | % collapsed) per lead window."""
    windows = tuple(args.windows)
    out_dir = PATHS.tables_dir(args.experiment, args.version, "decomposition")
    columns = [f"{tod} {metric}" for tod in DECOMP_TODS for metric in ("RMSE", "deb", "coll")]
    rows_by_window = {window: {} for window in windows}

    for key in REGION_DISPLAY:
        eval_name = REGION_EVAL_NAME.get(key, key)
        try:
            obs_full, predictions = load_entities(
                args.experiment, args.version, eval_name, args.variable, [], args.temporal
            )
        except FileNotFoundError as error:
            print(f"  [skip] {key} ({eval_name}): {error}")
            continue
        model_full = predictions["model"]
        for window in windows:
            obs_window = _select_window(obs_full, window)
            model_window = _select_window(model_full, window)
            rows_by_window[window][key] = _decomp_cells(model_window, obs_window)
        print(f"  {REGION_DISPLAY[key]:<14} done")

    for window in windows:
        stem = f"{args.experiment}_{args.version}_{window}"
        caption = (
            f"Per-pixel bias decomposition ({window}) for {args.experiment}: raw RMSE [K], "
            f"per-pixel-debiased RMSE [K], and \\% of MSE removed by debiasing, per time of day."
        )
        tex = latex_table(rows_by_window[window], columns, caption, f"tab:decomp_{stem}", REGION_DISPLAY)
        (out_dir / f"decomp_{stem}.tex").write_text(tex + "\n")
        pd.DataFrame.from_dict(rows_by_window[window], orient="index").to_csv(out_dir / f"decomp_{stem}.csv")
        print(f"[written] {out_dir / f'decomp_{stem}.tex'}")
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
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Per-region month x hour-of-day RMSE fingerprints (one panel per window) instead of the score tables.",
    )
    parser.add_argument(
        "--decomposition",
        action="store_true",
        help="Compact per-region bias-decomposition tables (raw | debiased RMSE | %% collapsed) instead of the score tables.",
    )
    parser.add_argument("--by-class", default="modis_lc_classes", help="Class coord pooled in --monthly class heatmap.")
    parser.add_argument(
        "--class-eval-name", default="train_mtg", help="Site whose pixels are pooled in --monthly class mode."
    )
    args = parser.parse_args()

    if args.monthly:
        run_monthly(args)
        return

    if args.fingerprint:
        run_fingerprint(args)
        return

    if args.decomposition:
        run_decomposition(args)
        return

    if args.regions is None or args.baselines is None:
        parser.error("--regions and --baselines are required unless --monthly/--fingerprint/--decomposition is set")

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
