import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.evaluation.conditioning import mask_time_of_day, time_of_day_mask


def _forecast_like(lon_values):
    init_times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lead_times = pd.timedelta_range("1h", periods=24, freq="1h")
    data = np.zeros((4, 24, len(lon_values)))
    return xr.DataArray(
        data,
        dims=("init_time", "lead_time", "lon"),
        coords={"init_time": init_times, "lead_time": lead_times, "lon": lon_values},
    )


def test_time_of_day_mask_at_lon_zero_matches_utc_window():
    da = _forecast_like([0.0])
    mask = time_of_day_mask(da, "day")

    init_hour = da["init_time"].dt.hour.values[:, None]
    lead_hour = (da["lead_time"].values / np.timedelta64(1, "h")).astype(int)[None, :]
    utc_hour = (init_hour + lead_hour) % 24
    expected = (utc_hour >= 10) & (utc_hour <= 15)
    np.testing.assert_array_equal(mask.isel(lon=0).values, expected)


def test_time_of_day_mask_shifts_with_longitude():
    da = _forecast_like([0.0, 45.0])  # 45 deg east = +3 h local solar
    mask = time_of_day_mask(da, "day")

    # The 00:00-init forecast reaches local hour 10 at lead 10h at lon=0,
    # but already at lead 7h at lon=45 (7 + 3 = 10).
    lead_hours = (da["lead_time"].values / np.timedelta64(1, "h")).astype(int)
    first_day_lead_lon0 = lead_hours[mask.isel(init_time=0, lon=0).values.argmax()]
    first_day_lead_lon45 = lead_hours[mask.isel(init_time=0, lon=1).values.argmax()]
    assert first_day_lead_lon0 == 10
    assert first_day_lead_lon45 == 7


def test_day_and_night_windows_are_disjoint():
    da = _forecast_like([0.0, 12.3, 45.0])
    day = time_of_day_mask(da, "day")
    night = time_of_day_mask(da, "night")
    assert not bool((day & night).any())


def test_mask_time_of_day_nans_outside_window():
    da = _forecast_like([0.0]) + 1.0  # no NaNs in the input
    masked = mask_time_of_day(da, "day")
    window = time_of_day_mask(da, "day")
    assert bool((masked.notnull() == window).all())
