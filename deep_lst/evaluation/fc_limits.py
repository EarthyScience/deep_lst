"""Forecast limit per (region, month, day/night) from daily-bucketed RMSE skill.

The forecast limit is the first forecast-day bucket whose pooled RMSE skill
(1 - RMSE_model / RMSE_baseline) drops below zero. Skill is the pooled average
(squared-error domain) over space and the month's init times -- no uncertainty.

Months are assigned by init_time (a forecast belongs to its launch month, so all
its leads stay in one month). Day/night is the local-solar-hour window (10-15 / 22-03),
reusing the table_runner definition. Outputs a tidy CSV (load for plotting) and two
LaTeX tables, daytime and nighttime, region x month (precompute for the paper).
"""

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.latex import write_month_tables
from deep_lst.evaluation.metrics import compute_pooled_by_day
from deep_lst.evaluation.table_runner import _mask_time_of_day, load_entities

MONTHS = list(range(1, 13))
DAYNIGHT = ("day", "night")


def _first_below_zero(days, values, horizon_day):
    """First forecast-day where skill drops below zero. Returns (limit_days, censored).

    censored=True when skill never drops below zero within the horizon (drawn at the
    horizon as an empty point). limit_days is None when there is no valid skill (no point).
    A crossing already at day 1 returns (day 1, False) -- a full point at day 1.
    """
    if not np.isfinite(values).any():
        return None, False
    below_zero = np.asarray(values) < 0  # NaN -> False, never a crossing
    crossing_idx = np.flatnonzero(below_zero)
    if crossing_idx.size == 0:
        return float(horizon_day), True
    return float(days[crossing_idx[0]]), False


def _skill_by_forecast_day(model, baseline, obs, month, kind):
    """Pooled RMSE skill (1 - model/baseline) per forecast-day bucket, for one init-month
    and one day/night window. Returns a DataArray on the 'day' dim (forecast-day number)."""
    init_months = pd.DatetimeIndex(model["init_time"].values).month
    in_month = init_months == month
    model_month = model.isel(init_time=in_month)
    baseline_month = baseline.isel(init_time=in_month)
    obs_month = _mask_time_of_day(obs.isel(init_time=in_month), kind)

    rmse_model = compute_pooled_by_day(model_month, obs_month, ["rmse"])["rmse"]
    rmse_baseline = compute_pooled_by_day(baseline_month, obs_month, ["rmse"])["rmse"]
    skill = 1.0 - rmse_model / rmse_baseline
    return skill.where(np.isfinite(rmse_baseline) & (rmse_baseline > 0))


def month_fc_limit(model, baseline, obs, month, kind):
    """Forecast limit (days) and censored flag for one (month, day/night) cell."""
    skill = _skill_by_forecast_day(model, baseline, obs, month, kind)
    if skill.sizes.get("day", 0) == 0:
        return None, False
    days = skill["day"].values
    return _first_below_zero(days, skill.values, days[-1])


def month_validity(obs, month, kind):
    """Clear-sky observation percentage for one (init-month, day/night) cell.

    Denominator = observable (ever-valid) land pixels inside the day/night window;
    numerator = the clear-sky observations among them. Returns None when the window
    is empty. The window mask reuses table_runner's day/night definition by masking a
    constant field, so the 10-15 / 22-03 bounds live in one place.
    """
    observable = obs.notnull().any(("init_time", "lead_time"))
    init_months = pd.DatetimeIndex(obs["init_time"].values).month
    obs_month = obs.isel(init_time=(init_months == month))
    window = _mask_time_of_day(xr.ones_like(obs_month), kind).notnull() & observable
    n_window = float(window.sum())
    if n_window == 0:
        return None
    n_clear = float(obs_month.where(window).notnull().sum())
    return 100.0 * n_clear / n_window


def build_records(experiment, version, region_specs, baseline, temporal, var="LST"):
    """Compute every (region, month, day/night) cell. region_specs: list of (label, eval_name).

    Returns tidy records: {region, eval_name, month, daynight, limit_days, censored}.
    """
    records = []
    for label, eval_name in region_specs:
        try:
            obs, predictions = load_entities(experiment, version, eval_name, var, [baseline], temporal)
        except FileNotFoundError as error:
            print(f"  [skip] {label} ({eval_name}): {error}")
            continue
        model = predictions["model"]
        base = predictions[baseline]
        latitude = float(obs["lat"].mean())
        for month in MONTHS:
            for kind in DAYNIGHT:
                limit_days, censored = month_fc_limit(model, base, obs, month, kind)
                records.append(
                    {
                        "region": label,
                        "eval_name": eval_name,
                        "latitude": latitude,
                        "month": month,
                        "daynight": kind,
                        "limit_days": limit_days,
                        "censored": censored,
                        "validity": month_validity(obs, month, kind),
                    }
                )
        print(f"  {label:<14} done")
    return records


def build_class_records(
    experiment, version, eval_name, class_key, baseline, class_labels, temporal, var="LST", min_pixels=20
):
    """Compute every (class, month, day/night) cell for one site, pooling pixels by `class_key`.

    Loads the site once with per-pixel class coords attached, then subsets pixels per class so each
    cell pools the squared errors of that class's pixels only. Classes with fewer than `min_pixels`
    pixels are skipped (too few to pool reliably). `class_labels` maps class id -> name. Returns tidy
    records with `region` set to the class label (so the table/heatmap helpers reuse).
    """
    obs, predictions = load_entities(experiment, version, eval_name, var, [baseline], temporal, attach_classes=True)
    model = predictions["model"]
    base = predictions[baseline]
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
        base_class = base.isel(pixel=pixel_mask)
        obs_class = obs.isel(pixel=pixel_mask)
        label = class_labels.get(str(int(class_id)), str(int(class_id)))
        for month in MONTHS:
            for kind in DAYNIGHT:
                limit_days, censored = month_fc_limit(model_class, base_class, obs_class, month, kind)
                records.append(
                    {
                        "region": label,
                        "class_id": int(class_id),
                        "eval_name": eval_name,
                        "month": month,
                        "daynight": kind,
                        "limit_days": limit_days,
                        "censored": censored,
                        "validity": month_validity(obs_class, month, kind),
                    }
                )
        print(f"  class {int(class_id):>3}  {label:<32} ({int(pixel_mask.sum())} px) done")
    return records


def write_csv(records, path):
    """Write the tidy records (one row per region/month/day-night) to CSV."""
    pd.DataFrame.from_records(records).to_csv(path, index=False)
    print(f"[written] {path}")


def _fc_cell(record):
    """Forecast-limit cell: integer days, '>=horizon' when censored, '--' when no data."""
    limit_days, censored = record["limit_days"], record["censored"]
    if limit_days is None:
        return "--"
    if censored:
        return f"$\\geq${limit_days:.0f}"
    return f"{limit_days:.0f}"


def _validity_cell(record):
    """Validity cell: clear-sky percentage, '--' when the window is empty."""
    validity = record["validity"]
    if validity is None:
        return "--"
    return f"{validity:.0f}\\%"


def write_latex_tables(records, region_order, out_dir, stem, caption_prefix, label_prefix):
    """Two region x month forecast-limit tables (daytime, nighttime)."""
    write_month_tables(records, region_order, out_dir, stem, caption_prefix, label_prefix, _fc_cell, MONTHS, DAYNIGHT)


def write_validity_tables(records, region_order, out_dir, stem, caption_prefix, label_prefix):
    """Two region x month clear-sky validity tables (daytime, nighttime)."""
    write_month_tables(
        records, region_order, out_dir, stem, caption_prefix, label_prefix, _validity_cell, MONTHS, DAYNIGHT
    )
