"""stats.amplitude_cells: the paired diurnal amplitude error on the jointly observed hours
with the coverage rule (the statistic only; verif_cell wires it to the cubes)."""

import numpy as np
import xarray as xr

from deep_lst.evaluation_climpred.stats import amplitude_cells

LEADS = np.arange(1, 49)  # two forecast days of a 00 UTC init at longitude 0: solar hour = lead % 24
MEAN = 300.0


def diurnal(amplitude):
    """Hourly cycle with its minimum at solar hour 4 and its maximum at solar hour 16."""
    return MEAN + amplitude / 2 * np.sin(2 * np.pi * (LEADS - 10) / 24)


def cube(values):
    coords = {"init": [np.datetime64("2024-03-01T00")], "lead": LEADS, "pixel": [0], "lon": ("pixel", [0.0])}
    return xr.DataArray(values[None, :, None], dims=("init", "lead", "pixel"), coords=coords)


def windows():
    hour = LEADS % 24
    peak = cube(((hour >= 12) & (hour < 16)).astype(float)) > 0
    trough = cube(((hour >= 3) & (hour < 7)).astype(float)) > 0
    return peak, trough


def observed_with(valid_hours):
    """Obs of amplitude 10 K, valid only at the given solar hours (same on both days)."""
    values = diurnal(10.0)
    keep = np.isin(LEADS % 24, valid_hours)
    return cube(np.where(keep, values, np.nan))


def test_complete_day_gives_the_exact_amplitude_difference():
    peak, trough = windows()
    out = amplitude_cells(cube(diurnal(14.0)), observed_with(range(24)), {}, peak, trough)
    assert out.sizes == {"init": 1, "day": 2, "pixel": 1}
    np.testing.assert_allclose(out["bias_model"].values, 4.0)
    np.testing.assert_allclose(out["amplitude_obs"].values, 10.0)
    np.testing.assert_allclose(out["amplitude_model_full"].values, 14.0)
    assert (out["n"].values == 24).all()
    assert out["peak_observed"].values.all() and out["trough_observed"].values.all()


def test_gaps_truncate_both_sides_alike():
    peak, trough = windows()
    valid_hours = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  # 12 hours with the extremes 4 and 16
    out = amplitude_cells(cube(diurnal(14.0)), observed_with(valid_hours), {}, peak, trough)
    np.testing.assert_allclose(out["bias_model"].values, 4.0)
    assert (out["n"].values == 12).all()


def test_missing_trough_window_fails_the_rule():
    peak, trough = windows()
    valid_hours = list(range(8, 24))  # 16 hours, afternoon seen, pre-dawn never
    out = amplitude_cells(cube(diurnal(14.0)), observed_with(valid_hours), {}, peak, trough)
    assert np.isnan(out["bias_model"].values).all()
    assert (out["n"].values == 0).all()
    assert out["peak_observed"].values.all() and not out["trough_observed"].values.any()


def test_too_few_hours_fail_the_rule():
    peak, trough = windows()
    out = amplitude_cells(cube(diurnal(14.0)), observed_with([4, 16, 10, 22]), {}, peak, trough)
    assert np.isnan(out["bias_model"].values).all()


def test_reference_scored_on_its_own_valid_hours():
    peak, trough = windows()
    clim = cube(diurnal(8.0))
    gappy = cube(np.where(np.isin(LEADS % 24, range(3, 7)), np.nan, diurnal(8.0)))  # no pre-dawn values
    references = {"clim": clim, "persistence": gappy}
    out = amplitude_cells(cube(diurnal(14.0)), observed_with(range(24)), references, peak, trough)
    np.testing.assert_allclose(out["mse_clim"].values, 4.0)  # (8 - 10)^2
    assert np.isnan(out["mse_persistence"].values).all()
    np.testing.assert_allclose(out["bias_model"].values, 4.0)  # the model is unaffected
