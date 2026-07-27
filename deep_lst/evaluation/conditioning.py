"""Conditioning layer for evaluation: named subsets of the forecast cube.

Convention: conditioning = boolean masks (or index selection for whole-axis
subsets) applied to pred/obs BEFORE statistics are accumulated. External
information enters as cube-aligned masks or label arrays; a label array plus
``isin`` converts an event list (days, classes) into a mask. Masks compose
with ``&``.
"""

import xarray as xr

from deep_lst.evaluation.metrics import local_valid_hour

__all__ = ["mask_time_of_day", "time_of_day_mask"]

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
