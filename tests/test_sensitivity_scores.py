"""sensitivity.scores: the hourly windows and the amplitude window on synthetic cubes."""

import numpy as np
import xarray as xr

from deep_lst.sensitivity.scores import AMPLITUDE_WINDOW, amplitude_errors, score_cube

LEADS = np.arange(1, 49)  # two forecast days of one 00 UTC init at longitude 0


def diurnal(amplitude):
    """Hourly cycle around 300 K, minimum at solar hour 4, maximum at solar hour 16."""
    return 300.0 + amplitude / 2 * np.sin(2 * np.pi * (LEADS - 10) / 24)


def cube(values):
    coords = {
        "init": [np.datetime64("2024-03-01T00")],
        "lead": LEADS,
        "pixel": [0, 1],
        "lon": ("pixel", [0.0, 0.0]),
        "lat": ("pixel", [10.0, 20.0]),
    }
    return xr.DataArray(
        np.broadcast_to(values[None, :, None], (1, 48, 2)).copy(), dims=("init", "lead", "pixel"), coords=coords
    )


def test_windows_and_amplitude_on_synthetic_cubes():
    reference = xr.Dataset({"LST": cube(diurnal(14.0))})
    observed = cube(diurnal(10.0))
    perturbed = cube(diurnal(12.0))
    reference_amplitude = amplitude_errors(reference["LST"], observed, reference)
    scores = score_cube(reference, observed, perturbed, ("all", "day", "night", AMPLITUDE_WINDOW), reference_amplitude)

    assert list(scores["window"].values) == ["all", "day", "night", AMPLITUDE_WINDOW]
    assert "lat" in scores.coords
    # hourly errors are 2 sin(.) for the reference and 1 sin(.) for the perturbed forecast
    np.testing.assert_allclose(scores.sel(window="all")["rmse_reference"].values, 2 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(scores.sel(window="all")["rmse_perturbed"].values, 1 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(scores.sel(window="all")["delta_rmse"].values, -1 / np.sqrt(2), rtol=1e-6)
    # amplitude errors 14 - 10 and 12 - 10 on every complete day
    amplitude = scores.sel(window=AMPLITUDE_WINDOW)
    np.testing.assert_allclose(amplitude["rmse_reference"].values, 4.0)
    np.testing.assert_allclose(amplitude["rmse_perturbed"].values, 2.0)
    assert (amplitude["n"].values == 1).all()
