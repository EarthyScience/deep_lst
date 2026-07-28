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
    measures_from_stats,
    pool_stats,
    skill_score,
    sufficient_stats,
    to_init_time,
    to_lead_hours,
)

__all__ = [
    "DIAG_PRODUCTS",
    "PRODUCTS",
    "ScoreProduct",
    "align_baseline",
    "compute_diagnostic_products",
    "compute_products",
    "compute_tables",
    "sample_stats",
]

RAW = ("rmse", "mae", "bias", "nse")


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


def required_tables(products: list) -> tuple:
    """Ordered unique (table, tod) keys covering every requested product."""
    needed = []
    for name in products:
        product = PRODUCTS[name]
        flavours = ("day", "night") if product.table == "map" else (product.tod,)
        for tod in flavours:
            if (product.table, tod) not in needed:
                needed.append((product.table, tod))
    return tuple(needed)


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

    `needed` is the output of `required_tables`. Profile tables reduce space and keep
    (init_time, lead_time); map tables keep space and pool (init, lead) under
    forecast-day labels per time-of-day flavour. anomaly tables require `clim`.
    """
    if window_source is None:
        window_source = pred
    space_dims = [d for d in pred.dims if d in ("lat", "lon", "pixel")]
    tables = {}
    for table, tod in needed:
        if table == "profile":
            pred_in, obs_in = _masked(pred, obs, tod, window_source)
        elif table == "anomaly_profile":
            if clim is None:
                raise ValueError("anomaly_profile requires a climate baseline cube")
            pred_in, obs_in = _masked(pred - clim, obs - clim, tod, window_source)
        elif table == "map":
            pred_in, obs_in = _masked(pred, obs, tod, window_source)
            stats = sufficient_stats(pred_in, obs_in, dim=["init_time"])
            tables[(table, tod)] = pool_stats(stats, by=forecast_day_labels(pred))
            continue
        else:
            raise ValueError(f"Unknown table kind: {table!r}")
        tables[(table, tod)] = sufficient_stats(pred_in, obs_in, dim=space_dims)
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
    needed = required_tables(products)
    tables = compute_tables(pred, obs, needed, clim=clim)
    baseline_tables = {}
    for baseline_name, baseline_cube in baselines.items():
        baseline_needed = tuple(key for key in needed if key[0] != "anomaly_profile")
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


# --- diagnostic products: the single-lead valid_time view ------------------------------
#
# At one lead the view is (valid_time, space), small enough to hold the tally at FULL
# granularity: the per-sample stats field. Every diagnostic table is then one pool_stats
# call on the same base — over space (per-timestep scores), over time by hour labels
# (diurnal), over time (per-pixel maps), over space by class labels (class violins),
# over everything by calendar-day labels (monthly violins).

DIAG_RAW = ("rmse", "mae", "bias", "nse")


@dataclass(frozen=True)
class DiagnosticProduct:
    """One single-lead product: which pooling of the per-sample stats, which measures."""

    measures: tuple
    pool: str  # "time" | "space" | "hour" | "class" | "day"
    anomaly: bool = False  # build from (pred - clim, obs - clim)
    with_baselines: bool = False  # emit {measure}_{name} and {measure}_skill_{name} columns
    with_means: bool = False  # emit pred_mean / obs_mean columns (from p_sum / o_sum)


DIAG_PRODUCTS: dict[str, DiagnosticProduct] = {
    # replaces legacy per-pixel scores (region_space) + plot_score_map / plot_improvement_map
    "diag_map": DiagnosticProduct(measures=DIAG_RAW, pool="time", with_baselines=True),
    "diag_anomaly_map": DiagnosticProduct(measures=("acc",), pool="time", anomaly=True),
    # today: plot_error_overview's region_time scores
    "diag_scores_by_time": DiagnosticProduct(measures=DIAG_RAW, pool="space", with_baselines=True),
    "diag_anomaly_by_time": DiagnosticProduct(measures=("acc", "activity"), pool="space", anomaly=True),
    # today: grouped / climate_grouped (diurnal, region_space) incl. pred_mean/obs_mean
    "diag_by_hour": DiagnosticProduct(measures=DIAG_RAW, pool="hour", with_baselines=True, with_means=True),
    "diag_anomaly_by_hour": DiagnosticProduct(measures=("acc", "activity"), pool="hour", anomaly=True, with_means=True),
    # replaces legacy class scores (expanded per class coord present)
    "diag_by_class": DiagnosticProduct(measures=DIAG_RAW, pool="class"),
    "diag_anomaly_by_class": DiagnosticProduct(measures=("acc", "activity"), pool="class", anomaly=True),
    # today: daily_pooled / daily_pooled_anom (monthly violins; adds {measure}_total)
    "diag_by_day": DiagnosticProduct(measures=DIAG_RAW, pool="day"),
    "diag_anomaly_by_day": DiagnosticProduct(measures=("acc", "activity"), pool="day", anomaly=True),
}


def sample_stats(pred: xr.DataArray, obs: xr.DataArray) -> xr.Dataset:
    """Per-sample sufficient statistics (full tally): the unreduced base of every diagnostic table.

    All variables are zero-filled outside the jointly valid samples, so any later
    pooling is a plain sum (`pool_stats` does not skip NaNs -- there are none).
    """
    valid = pred.notnull() & obs.notnull()
    error = (pred - obs).where(valid)
    obs_valid = obs.where(valid).astype("float64")
    pred_valid = pred.where(valid).astype("float64")

    return xr.Dataset(
        {
            "n": valid.astype("int64"),
            "e_sum": error.fillna(0.0),
            "se_sum": (error**2).fillna(0.0),
            "ae_sum": np.abs(error).fillna(0.0),
            "o_sum": obs_valid.fillna(0.0),
            "o2_sum": (obs_valid**2).fillna(0.0),
            "p_sum": pred_valid.fillna(0.0),
            "p2_sum": (pred_valid**2).fillna(0.0),
            "po_sum": (pred_valid * obs_valid).fillna(0.0),
        }
    )


def _pool_diag(base: xr.Dataset, pool: str, time_dim: str, space_dims: list) -> xr.Dataset:
    from deep_lst.evaluation.conditioning import calendar_day_labels, hour_of_day_labels, roll_hours_to_solar

    if pool == "time":
        return pool_stats(base, dim=[time_dim])
    if pool == "space":
        return pool_stats(base, dim=space_dims)
    if pool == "hour":
        pooled = pool_stats(base, by=hour_of_day_labels(base, time_dim))
        # keep the full 24-hour axis: absent hours become NaN rows, matching the legacy
        # grouped output (a 6-hourly-init single lead only populates 4 of 24 bins);
        # then relabel to local solar hours per longitude (exact roll for hourly data)
        return roll_hours_to_solar(pooled.reindex(hour=np.arange(24)))
    if pool == "day":
        pooled = pool_stats(base, dim=space_dims, by=calendar_day_labels(base, time_dim))
        months = pd.DatetimeIndex(pooled["day"].values).month
        return pooled.assign_coords(month=("day", months))
    raise ValueError(f"Unknown diagnostic pooling: {pool!r}")


def _finish_diag(pooled: xr.Dataset, product: DiagnosticProduct) -> xr.Dataset:
    finished = measures_from_stats(pooled, product.measures)
    if product.with_means:
        finished["pred_mean"] = pooled["p_sum"] / pooled["n"]
        finished["obs_mean"] = pooled["o_sum"] / pooled["n"]
    if product.pool == "day":
        totals = measures_from_stats(pool_stats(pooled, dim=["day"]), product.measures)
        for measure in product.measures:
            finished[f"{measure}_total"] = totals[measure]
        finished = finished.assign_coords(month=pooled["month"])
    return finished


def compute_diagnostic_products(
    pred: xr.DataArray,
    obs: xr.DataArray,
    products: list,
    baselines: dict[str, xr.DataArray] | None = None,
    clim: xr.DataArray | None = None,
    class_coords: tuple = (),
    tod: str = "all",
) -> dict[str, xr.Dataset]:
    """Evaluate DIAG_PRODUCTS on the single-lead (valid_time, space) view.

    `tod` conditions EVERY product on the local-solar day/night window by masking the
    inputs before the per-sample stats (the window depends on lon, so it must apply
    while space still exists). "diag_by_class" expands to one output per class coord
    present on `pred`, keyed ``diag_by_class__{coord}``.
    """
    baselines = baselines or {}
    if tod != "all":
        window = time_of_day_mask(pred, tod)
        pred = pred.where(window)
        obs = obs.where(window)
        baselines = {name: cube.where(window) for name, cube in baselines.items()}
        if clim is not None:
            clim = clim.where(window)

    time_dim = next(d for d in pred.dims if d in ("valid_time", "init_time"))
    space_dims = [d for d in pred.dims if d in ("lat", "lon", "pixel")]

    requested = [DIAG_PRODUCTS[name] for name in products]
    base = sample_stats(pred, obs)
    anomaly_base = None
    if any(p.anomaly for p in requested):
        if clim is None:
            raise ValueError("anomaly diagnostic products require a climate baseline cube")
        anomaly_base = sample_stats(pred - clim, obs - clim)
    baseline_bases = {name: sample_stats(cube, obs) for name, cube in baselines.items()}

    out = {}
    for name, product in zip(products, requested):
        source = anomaly_base if product.anomaly else base

        if product.pool == "class":
            for coord in class_coords:
                if coord not in pred.coords:
                    continue
                pooled = pool_stats(source, by=pred[coord].rename(coord))
                pooled = pooled.sel({coord: pooled[coord] != -1})  # -1 = no class
                out[f"{name}__{coord}"] = measures_from_stats(pooled, product.measures)
            continue

        pooled = _pool_diag(source, product.pool, time_dim, space_dims)
        finished = _finish_diag(pooled, product)
        if product.with_baselines:
            for bl_name, bl_base in baseline_bases.items():
                bl_pooled = _pool_diag(bl_base, product.pool, time_dim, space_dims)
                bl_finished = measures_from_stats(bl_pooled, product.measures)
                for measure in product.measures:
                    finished[f"{measure}_{bl_name}"] = bl_finished[measure]
                    finished[f"{measure}_skill_{bl_name}"] = skill_score(finished[measure], bl_finished[measure])
        out[name] = finished
    return out


def align_baseline(baseline: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Align a valid_time-indexed baseline to pred's (init_time, lead_time, space) cube.

    Per lead: reindex to init + lead valid times, relabel back to init_time, mask to
    finite pred (the same per-lead alignment the legacy evaluator performed).
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
