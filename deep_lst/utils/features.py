"""Computed input features that are not stored in any modality's eozarr.

eodata v2 dropped the on-the-fly solar/time variables, so deep_lst computes them here from
the anchor hour (T1H global index) and, for the solar features, the cell lon/lat:

- ``doy_cos/doy_sin`` — cyclic day-of-year encoding. Global (same value everywhere at a given
  hour), so attached on the anchor/seviri grid.
- ``solar_zenith`` — solar zenith angle in degrees at a cell's lon/lat.
- ``hod_cos/hod_sin`` — cyclic local solar hour encoding: cos/sin of the solar hour angle at the
  cell's longitude, so (1, 0) is local solar noon and hod_sin < 0 before noon.
  Both solar features are per-grid: on the coarse modalities they carry the geometric offset of
  the forcing cell relative to the target pixel.

Hour indices are converted to timestamps via eodata's ``T1H`` grid (single source of truth for
the index<->date mapping), so this stays consistent with how the stores are indexed.
"""

import numpy as np
from eodata.grids import T1H


def _timestamps(hours):
    """T1H global hour indices -> pandas DatetimeIndex (UTC)."""
    return T1H.index_to_startdate(np.asarray(hours, dtype=np.int64))


def time_features(hours):
    """{doy_cos, doy_sin} for T1H global hour indices (each shape == hours).

    The circle closes exactly every year: Jan 1 -> 0, Dec 31 -> 2*pi*(N-1)/N.
    """
    timestamps = _timestamps(hours)
    year_length = 365.0 + np.asarray(timestamps.is_leap_year)  # DatetimeIndex gives a plain ndarray
    doy_angle = 2.0 * np.pi * (timestamps.dayofyear.to_numpy() - 1) / year_length
    return {
        "doy_cos": np.cos(doy_angle),
        "doy_sin": np.sin(doy_angle),
    }


def solar_features(lat, lon, hours):
    """{solar_zenith, hod_cos, hod_sin} at (lat, lon) for T1H global hour indices (NOAA formulas).

    solar_zenith is in degrees. hod_cos/hod_sin encode the local solar hour angle at this
    longitude (zero at solar noon, negative before), so they replace a UTC hour-of-day encoding.
    """
    timestamps = _timestamps(hours)
    year_length = 365.0 + np.asarray(timestamps.is_leap_year)  # DatetimeIndex gives a plain ndarray
    day = timestamps.dayofyear.to_numpy()
    frac_hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    # Fractional year (radians), then equation of time (minutes) and solar declination (radians).
    gamma = 2.0 * np.pi / year_length * (day - 1 + (frac_hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    # True solar time -> hour angle (times are UTC, so the longitude term is the only offset).
    true_solar_time = frac_hour * 60.0 + eqtime + 4.0 * lon
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)
    lat_rad = np.deg2rad(lat)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    return {
        "solar_zenith": np.rad2deg(np.arccos(np.clip(cos_zenith, -1.0, 1.0))),
        "hod_cos": np.cos(hour_angle),
        "hod_sin": np.sin(hour_angle),
    }
