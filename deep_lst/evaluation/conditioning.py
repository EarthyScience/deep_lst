"""Conditioning layer for evaluation: named subsets of the forecast cube.

Convention: conditioning = boolean masks (or index selection for whole-axis
subsets) applied to pred/obs BEFORE statistics are accumulated. External
information enters as cube-aligned masks or label arrays; a label array plus
``isin`` converts an event list (days, classes) into a mask. Masks compose
with ``&``.
"""

import numpy as np
import xarray as xr

from deep_lst.evaluation.metrics import LEAD_DAY_BUCKETS, local_valid_hour, to_lead_hours

__all__ = [
    "forecast_day_labels",
    "mask_time_of_day",
    "roll_hours_to_solar",
    "solar_hour_shift",
    "time_of_day_mask",
    "valid_day_labels",
]

DAY_HOURS = (10, 15)
NIGHT_HOURS = (22, 3)


def time_of_day_mask(da: xr.DataArray, kind: str) -> xr.DataArray:
    """Boolean day/night mask on the LOCAL solar valid hour (day 10-15, night 22-03).

    Broadcasts over init_time, lead_time and whatever dim carries ``lon``
    (the grid axis, or ``pixel`` for sparse sites) — see `local_valid_hour`.
    """
    valid_hour = local_valid_hour(da)
    if kind == "day":
        return (valid_hour >= DAY_HOURS[0]) & (valid_hour <= DAY_HOURS[1])
    if kind == "night":
        return (valid_hour >= NIGHT_HOURS[0]) | (valid_hour <= NIGHT_HOURS[1])
    raise ValueError(f"Unknown time-of-day kind: {kind!r}")


def mask_time_of_day(da: xr.DataArray, kind: str) -> xr.DataArray:
    """``da`` with everything outside the ``kind`` window set to NaN."""
    return da.where(time_of_day_mask(da, kind))


def forecast_day_labels(da: xr.DataArray) -> xr.DataArray:
    """Forecast day (1-10) of each lead from `LEAD_DAY_BUCKETS`; 0 for leads outside all buckets."""
    lead_hours = to_lead_hours(da)
    day_of_lead = np.zeros(lead_hours.size, dtype=int)
    for day_number, (low, high) in enumerate(LEAD_DAY_BUCKETS, start=1):
        in_bucket = (lead_hours >= low) & (lead_hours <= high)
        day_of_lead[in_bucket] = day_number
    return xr.DataArray(day_of_lead, dims="lead_time", name="day")


def valid_day_labels(obj) -> xr.DataArray:
    """Valid calendar day (init + lead, floored to days) on ``(init_time, lead_time)``."""
    init = obj["init_time"].values[:, None]
    lead = obj["lead_time"].values[None, :]
    day = (init + lead).astype("datetime64[D]")
    return xr.DataArray(day, dims=("init_time", "lead_time"), name="day")


def hour_of_day_labels(obj, time_dim: str = "valid_time") -> xr.DataArray:
    """UTC hour-of-day (0-23) of each timestamp on a 1-D time dim.

    The UTC groupby step of the diurnal axis; `roll_hours_to_solar` relabels the
    pooled result to local solar hours afterwards (pooling by UTC hour first keeps
    the space dims, which a longitude-dependent label array would stack away).
    """
    hours = obj[time_dim].dt.hour.values
    return xr.DataArray(hours, dims=time_dim, name="hour")


def solar_hour_shift(obj) -> xr.DataArray:
    """Whole-hour offset from UTC to local solar time per longitude, ``round(lon / 15)``.

    Hourly data sits on integer UTC hours, so every sample's nearest-integer solar hour
    is its UTC hour plus this constant per-longitude shift — switching an hour axis to
    solar time is then a pure relabelling per longitude, never a re-pooling.
    """
    shift = np.round(obj["lon"].values / 15.0).astype(int)
    lon_dim = "lon" if "lon" in obj.dims else "pixel"
    return xr.DataArray(shift, dims=lon_dim, name="solar_shift")


def roll_hours_to_solar(pooled: xr.Dataset) -> xr.Dataset:
    """Relabel a UTC ``hour`` axis (0-23) to local solar hours, per longitude.

    ``pooled`` carries an ``hour`` dim and a ``lon`` coordinate (the grid axis, or
    per-pixel for sparse sites). Solar bin h at a longitude with shift s holds the
    UTC bin (h - s) % 24 — exact for hourly data (see `solar_hour_shift`). Hours a
    pixel never observes stay NaN, at their solar position.
    """
    shift = solar_hour_shift(pooled)
    hours = pooled["hour"].values.astype(int)
    source = (hours[:, None] - shift.values[None, :]) % 24
    indexer = xr.DataArray(source, dims=("hour", shift.dims[0]))
    rolled = pooled.isel(hour=indexer)
    return rolled.assign_coords(hour=("hour", hours))


def calendar_day_labels(obj, time_dim: str = "valid_time") -> xr.DataArray:
    """Calendar day of each timestamp on a 1-D time dim (for daily pooling of a single-lead view)."""
    days = obj[time_dim].dt.floor("D").values
    return xr.DataArray(days, dims=time_dim, name="day")
