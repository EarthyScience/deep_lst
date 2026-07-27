"""Declarative evaluation products on shared sufficient-statistics tables.

A product = (table, conditioning, pooling, measures), declared once in PRODUCTS.
The expensive act is building each stats TABLE exactly once per evaluation:

  profile          sufficient_stats reduced over space -> (init_time, lead_time);
                   flavours "all"/"day"/"night" (the time-of-day mask depends on
                   longitude, so it is applied BEFORE the space reduction)
  map              per-pixel stats pooled over (init, lead) under forecast-day
                   labels x time-of-day -> (space, day, tod)
  anomaly_profile  profile built from (pred - clim, obs - clim) with the
                   ACC/activity cross-terms

Every product is then a cheap pool_stats + measures_from_stats on its table;
`with_skill` products divide against the same table built for a baseline cube.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.conditioning import forecast_day_labels, time_of_day_mask, valid_day_labels
from deep_lst.evaluation.metrics import (
    MEASURE_REQUIRES,
    measures_from_stats,
    pool_stats,
    skill_score,
    sufficient_stats,
    to_init_time,
    to_lead_hours,
)

__all__ = ["PRODUCTS", "ScoreProduct", "align_baseline", "compute_products", "compute_tables"]

RAW = ("rmse", "mae", "bias", "nse")
_BASE_STATS = ("n", "e_sum", "se_sum")


@dataclass(frozen=True)
class ScoreProduct:
    """One named evaluation product: which table, conditioned how, pooled how, which measures."""

    table: str  # "profile" | "map" | "anomaly_profile"
    measures: tuple
    tod: str = "all"  # profile flavour ("all" | "day" | "night"); ignored for "map"
    by: tuple = ()  # label builders applied when pooling: "forecast_day" | "valid_day"
    lead_window: tuple | None = None  # inclusive lead-hour bounds, e.g. (1, 24)
    pool_dims: tuple = ()  # plain-summed dims, e.g. ("init_time",)
    cumulative_days: bool = False  # map only: pool day buckets cumulatively (<= Day N)
    with_skill: bool = False  # also emit each measure vs every baseline table
    keep_table_dims: bool = field(default=False)  # finish the table directly (no pooling)


PRODUCTS: dict[str, ScoreProduct] = {
    # verification curves (today: scores / skill_scores on (init, lead))
    "scores_by_init_lead": ScoreProduct(table="profile", measures=RAW, keep_table_dims=True, with_skill=True),
    # today: pooled_scores
    "scores_by_lead": ScoreProduct(table="profile", measures=RAW, pool_dims=("init_time",), with_skill=True),
    # today: pooled_scores_day + pooled_skill_day
    "scores_by_forecast_day": ScoreProduct(
        table="profile", measures=RAW, pool_dims=("init_time",), by=("forecast_day",), with_skill=True
    ),
    # day/night error growth (today computed inside skill_day_night_plot)
    "rmse_by_forecast_day_daytime": ScoreProduct(
        table="profile", measures=("rmse",), tod="day", pool_dims=("init_time",), by=("forecast_day",), with_skill=True
    ),
    "rmse_by_forecast_day_nighttime": ScoreProduct(
        table="profile",
        measures=("rmse",),
        tod="night",
        pool_dims=("init_time",),
        by=("forecast_day",),
        with_skill=True,
    ),
    # today: daily_pooled_bucket ("Day 1" / "Day 10" valid-day violins)
    "rmse_by_valid_day_lead1_24": ScoreProduct(
        table="profile", measures=("rmse",), lead_window=(1, 24), by=("valid_day",)
    ),
    "rmse_by_valid_day_lead217_240": ScoreProduct(
        table="profile", measures=("rmse",), lead_window=(217, 240), by=("valid_day",)
    ),
    # today: climate_scores (per init x lead) and pooled_climate_day
    "anomaly_by_init_lead": ScoreProduct(table="anomaly_profile", measures=("acc", "activity"), keep_table_dims=True),
    "anomaly_by_forecast_day": ScoreProduct(
        table="anomaly_profile", measures=("acc", "activity"), pool_dims=("init_time",), by=("forecast_day",)
    ),
    # today: acc per day/night bucket inside acc_day_night_plot
    "anomaly_by_forecast_day_daytime": ScoreProduct(
        table="anomaly_profile", measures=("acc",), tod="day", pool_dims=("init_time",), by=("forecast_day",)
    ),
    "anomaly_by_forecast_day_nighttime": ScoreProduct(
        table="anomaly_profile", measures=("acc",), tod="night", pool_dims=("init_time",), by=("forecast_day",)
    ),
    # today: _bucket_tod_rmse inside plot_improvement_skill_maps (cumulative <= Day N buckets)
    "rmse_map_by_bucket_tod": ScoreProduct(table="map", measures=("rmse",), cumulative_days=True, with_skill=True),
}


def required_stats(products: list) -> dict:
    """{(table, tod): extras tuple} covering every requested product."""
    needed = {}
    for name in products:
        product = PRODUCTS[name]
        flavours = ("day", "night") if product.table == "map" else (product.tod,)
        for tod in flavours:
            key = (product.table, tod)
            extras = {
                column
                for measure in product.measures
                for column in MEASURE_REQUIRES[measure]
                if column not in _BASE_STATS
            }
            needed[key] = tuple(sorted(set(needed.get(key, ())) | extras))
    return needed


def _masked(pred: xr.DataArray, obs: xr.DataArray, tod: str, window_source: xr.DataArray):
    if tod == "all":
        return pred, obs
    # The window is always built from the MODEL pred: aligned baseline cubes and
    # anomaly differences can lose the lon coordinate local_valid_hour needs.
    window = time_of_day_mask(window_source, tod)
    return pred.where(window), obs.where(window)


def compute_tables(
    pred: xr.DataArray,
    obs: xr.DataArray,
    needed: dict,
    clim: xr.DataArray | None = None,
    window_source: xr.DataArray | None = None,
) -> dict:
    """Build each required stats table exactly once.

    `needed` is the output of `required_stats`. Profile tables reduce space and keep
    (init_time, lead_time); map tables keep space and pool (init, lead) under
    forecast-day labels per time-of-day flavour. anomaly tables require `clim`.
    """
    if window_source is None:
        window_source = pred
    space_dims = [d for d in pred.dims if d in ("lat", "lon", "pixel")]
    tables = {}
    for (table, tod), extras in needed.items():
        if table == "profile":
            pred_in, obs_in = _masked(pred, obs, tod, window_source)
        elif table == "anomaly_profile":
            if clim is None:
                raise ValueError("anomaly_profile requires a climate baseline cube")
            pred_in, obs_in = _masked(pred - clim, obs - clim, tod, window_source)
        elif table == "map":
            pred_in, obs_in = _masked(pred, obs, tod, window_source)
            stats = sufficient_stats(pred_in, obs_in, dim=["init_time"], extra=extras)
            tables[(table, tod)] = pool_stats(stats, by=forecast_day_labels(pred))
            continue
        else:
            raise ValueError(f"Unknown table kind: {table!r}")
        tables[(table, tod)] = sufficient_stats(pred_in, obs_in, dim=space_dims, extra=extras)
    return tables


def _lead_subset(stats: xr.Dataset, lead_window: tuple) -> xr.Dataset:
    low, high = lead_window
    lead_hours = to_lead_hours(stats)
    keep = np.where((lead_hours >= low) & (lead_hours <= high))[0]
    return stats.isel(lead_time=keep)


def _pool_product(stats: xr.Dataset, product: ScoreProduct) -> xr.Dataset:
    if product.keep_table_dims:
        return stats
    if product.lead_window is not None:
        stats = _lead_subset(stats, product.lead_window)

    labels = None
    if product.by == ("forecast_day",):
        labels = forecast_day_labels(stats)
    elif product.by == ("valid_day",):
        labels = valid_day_labels(stats)
    elif product.by:
        raise ValueError(f"Unknown label builders: {product.by!r}")

    pooled = pool_stats(stats, dim=list(product.pool_dims) or None, by=labels)
    if product.by == ("forecast_day",):
        pooled = pooled.sel(day=pooled["day"] != 0)  # leads outside every bucket
        pooled = pooled.assign_coords(day_label=("day", [f"Day {d}" for d in pooled["day"].values]))
    if product.by == ("valid_day",):
        # Fully-cloudy days carry no information. The counts must be materialised first:
        # boolean-dask indexers are rejected by xarray, and after the day-groupby they
        # are a tiny 1-D array anyway.
        counts = pooled["n"].load()
        pooled = pooled.where(counts > 0, drop=True)
        months = pd.DatetimeIndex(pooled["day"].values).month
        pooled = pooled.assign_coords(month=("day", months))
    return pooled


def _cumulative_days(stats: xr.Dataset) -> xr.Dataset:
    """Map-table day buckets pooled cumulatively: entry N = all buckets <= Day N (sums add)."""
    cumulative = stats.cumsum("day")
    cumulative = cumulative.assign_coords(day=stats["day"].values)
    return cumulative.assign_coords(day_label=("day", [f"≤ Day {d}" for d in cumulative["day"].values]))


def compute_products(
    pred: xr.DataArray,
    obs: xr.DataArray,
    products: list,
    baselines: dict[str, xr.DataArray] | None = None,
    clim: xr.DataArray | None = None,
) -> dict[str, xr.Dataset]:
    """Evaluate the named PRODUCTS entries: tables once, then pool + finish per product.

    `baselines` maps name -> aligned (init_time, lead_time, space) cube (see
    `align_baseline`); `with_skill` products gain ``{measure}_skill_{name}`` variables.
    `clim` is the aligned climate cube used for anomaly tables.
    """
    baselines = baselines or {}
    needed = required_stats(products)
    tables = compute_tables(pred, obs, needed, clim=clim)
    baseline_tables = {}
    for baseline_name, baseline_cube in baselines.items():
        baseline_needed = {key: extras for key, extras in needed.items() if key[0] != "anomaly_profile"}
        baseline_tables[baseline_name] = compute_tables(
            baseline_cube, obs, baseline_needed, clim=None, window_source=pred
        )

    out = {}
    for name in products:
        product = PRODUCTS[name]
        if product.table == "map":
            tod_slices = []
            for tod in ("day", "night"):
                tod_stats = tables[(product.table, tod)]
                if product.cumulative_days:
                    tod_stats = _cumulative_days(tod_stats)
                tod_slices.append(tod_stats)
            stats = xr.concat(tod_slices, dim=pd.Index(["day", "night"], name="tod"))
            finished = measures_from_stats(stats, product.measures)
            if product.with_skill:
                for baseline_name, per_table in baseline_tables.items():
                    bl_slices = []
                    for tod in ("day", "night"):
                        bl_stats = per_table[(product.table, tod)]
                        if product.cumulative_days:
                            bl_stats = _cumulative_days(bl_stats)
                        bl_slices.append(bl_stats)
                    bl_stats = xr.concat(bl_slices, dim=pd.Index(["day", "night"], name="tod"))
                    bl_finished = measures_from_stats(bl_stats, product.measures)
                    for measure in product.measures:
                        finished[f"{measure}_{baseline_name}"] = bl_finished[measure]
                        finished[f"{measure}_skill_{baseline_name}"] = skill_score(
                            finished[measure], bl_finished[measure]
                        )
            out[name] = finished
            continue

        stats = _pool_product(tables[(product.table, product.tod)], product)
        finished = measures_from_stats(stats, product.measures)
        if product.by == ("valid_day",):
            # The monthly violins draw the bucket-pooled total as a reference line.
            total_stats = tables[(product.table, product.tod)]
            if product.lead_window is not None:
                total_stats = _lead_subset(total_stats, product.lead_window)
            totals = measures_from_stats(pool_stats(total_stats, dim=["init_time", "lead_time"]), product.measures)
            for measure in product.measures:
                finished[f"{measure}_total"] = totals[measure]
        if product.with_skill:
            for baseline_name, per_table in baseline_tables.items():
                bl_stats = _pool_product(per_table[(product.table, product.tod)], product)
                bl_finished = measures_from_stats(bl_stats, product.measures)
                for measure in product.measures:
                    finished[f"{measure}_{baseline_name}"] = bl_finished[measure]
                    finished[f"{measure}_skill_{baseline_name}"] = skill_score(finished[measure], bl_finished[measure])
        out[name] = finished
    return out


def align_baseline(baseline: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Align a valid_time-indexed baseline to pred's (init_time, lead_time, space) cube.

    Per lead: reindex to init + lead valid times, relabel back to init_time, mask to
    finite pred (the same alignment EvaluatorVerification._baseline_full performs).
    """
    slices = []
    lead_times = pred["lead_time"].values
    for lead_time in lead_times:
        lead_hours = int(pd.Timedelta(lead_time).total_seconds() // 3600)
        pred_lead = pred.sel(lead_time=lead_time)
        valid_times = pred_lead["init_time"].values + np.timedelta64(lead_hours, "h")
        aligned = baseline.reindex(valid_time=valid_times)
        aligned = to_init_time(aligned, f"{lead_hours}h")
        slices.append(aligned.where(np.isfinite(pred_lead)))
    return xr.concat(slices, dim=pd.Index(lead_times, name="lead_time")).transpose(*pred.dims)
