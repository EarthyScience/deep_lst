"""sensitivity.scores: the hourly windows and the amplitude window on synthetic cubes, with a
perturbed store that keeps a thinned init set (the CPU launcher scores 00 UTC inits only)."""

import numpy as np
import xarray as xr

from deep_lst.sensitivity.scores import AMPLITUDE_WINDOW, amplitude_errors, score_cube

LEADS = np.arange(1, 49)  # two forecast days per init, 00 UTC inits at longitude 0
INITS = [np.datetime64("2024-03-01T00"), np.datetime64("2024-03-02T00")]


def diurnal(amplitude):
    """Hourly cycle around 300 K, minimum at solar hour 4, maximum at solar hour 16."""
    return 300.0 + amplitude / 2 * np.sin(2 * np.pi * (LEADS - 10) / 24)


def cube(values, inits=INITS):
    coords = {
        "init": inits,
        "lead": LEADS,
        "pixel": [0, 1],
        "lon": ("pixel", [0.0, 0.0]),
        "lat": ("pixel", [10.0, 20.0]),
    }
    data = np.broadcast_to(values[None, :, None], (len(inits), 48, 2)).copy()
    return xr.DataArray(data, dims=("init", "lead", "pixel"), coords=coords)


def test_windows_and_amplitude_with_a_thinned_perturbed_store():
    reference = xr.Dataset({"LST": cube(diurnal(14.0))})  # two inits, like forecast.zarr
    observed = cube(diurnal(10.0))
    perturbed = cube(diurnal(12.0), inits=INITS[:1])  # the perturbed store keeps one init only
    reference_amplitude = amplitude_errors(reference["LST"], observed)
    windows = ("all", "day", "night", AMPLITUDE_WINDOW)
    scores = score_cube(reference, observed, perturbed, windows, reference_amplitude)

    assert list(scores["window"].values) == list(windows)
    assert "lat" in scores.coords
    # hourly errors are 2 sin(.) for the reference and 1 sin(.) for the perturbed forecast,
    # on the one shared init: 24 joint hours per forecast day
    hourly = scores.sel(window="all")
    assert (hourly["n"].values == 24).all()
    np.testing.assert_allclose(hourly["rmse_reference"].values, 2 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(hourly["rmse_perturbed"].values, 1 / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(hourly["delta_rmse"].values, -1 / np.sqrt(2), rtol=1e-6)
    # amplitude errors 14 - 10 and 12 - 10 on every complete day of the shared init
    amplitude = scores.sel(window=AMPLITUDE_WINDOW)
    assert (amplitude["n"].values == 1).all()
    np.testing.assert_allclose(amplitude["rmse_reference"].values, 4.0)
    np.testing.assert_allclose(amplitude["rmse_perturbed"].values, 2.0)
