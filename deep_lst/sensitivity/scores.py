"""RMSE change per pixel and forecast day when one input is perturbed, against the run's own
forecast.zarr, on the samples where both forecasts and the observation are valid.

The windows all/day/night pool the hourly errors of the solar-hour masks. The window
"amplitude" is the diurnal amplitude error of every forecast day, with the estimator and
coverage rule of the per-cell verification (evaluation_climpred.stats.amplitude_cells, as in
verif_cell): forecast minus obs amplitude over the jointly observed hours of the day, kept when
at least MIN_OBSERVED_HOURS joint hours include one in the peak and one in the trough window.
Both forecasts exist on the same clear-init days, so every comparison is paired; in that window
the RMSE columns carry the RMS amplitude error.

Writes scores.nc (variable, window, day, pixel: rmse_reference, rmse_perturbed, delta_rmse, n)
and summary.csv (per variable, window, day: pooled RMSE of both, pooled delta, median pixel
delta, share of pixels with a positive delta) into the sensitivity folder.
"""

import numpy as np
import xarray as xr

from deep_lst.evaluation_climpred.cubes import obs_cube, obs_series, prepare_initialised
from deep_lst.evaluation_climpred.masks import MASKS, solar_hour_mask
from deep_lst.evaluation_climpred.stats import amplitude_cells
from deep_lst.evaluation_climpred.verif_cell import AMPLITUDE_WINDOW, PEAK_HOURS, TROUGH_HOURS

WINDOWS = ("all", "day", "night", AMPLITUDE_WINDOW)


def pixel_day_rmse(cube, observed, keep, day):
    """(day, pixel) RMSE of one forecast cube over the kept (init, lead) samples."""
    squared = ((cube - observed) ** 2).where(keep)
    total = squared.groupby(day).sum("lead").sum("init")
    count = keep.groupby(day).sum("lead").sum("init")
    return np.sqrt(total / count), count


def amplitude_errors(cube, observed, reference):
    """Signed diurnal amplitude error per (init, day, pixel) of one forecast cube; the solar-hour
    windows of the rule come from the reference cube's init times and longitudes."""
    peak = solar_hour_mask(PEAK_HOURS)(reference)
    trough = solar_hour_mask(TROUGH_HOURS)(reference)
    return amplitude_cells(cube, observed, {}, peak, trough)["bias_model"]


def amplitude_window(reference_errors, perturbed_errors):
    """(day, pixel) RMS amplitude error of both forecasts over the inits where both days count."""
    keep = reference_errors.notnull() & perturbed_errors.notnull()
    count = keep.sum("init")
    rmse_reference = np.sqrt((reference_errors**2).where(keep).sum("init") / count)
    rmse_perturbed = np.sqrt((perturbed_errors**2).where(keep).sum("init") / count)
    return xr.Dataset({"rmse_reference": rmse_reference, "rmse_perturbed": rmse_perturbed, "n": count})


def score_cube(reference, observed, perturbed, windows, reference_amplitude=None):
    """Per-window (day, pixel) RMSE of the reference and the perturbed forecast on their joint
    samples; reference_amplitude = amplitude_errors of the reference, needed for the amplitude window."""
    valid = reference["LST"].notnull() & observed.notnull() & perturbed.notnull()
    day = xr.DataArray((reference["lead"].values - 1) // 24 + 1, dims="lead", name="day")
    pixel_coords = {name: coord for name, coord in reference.coords.items() if set(coord.dims) == {"pixel"}}
    per_window = []
    for window in windows:
        if window == AMPLITUDE_WINDOW:
            scores = amplitude_window(reference_amplitude, amplitude_errors(perturbed, observed, reference))
            per_window.append(scores.assign_coords(pixel_coords))
            continue
        mask_fn = MASKS[window]
        keep = valid if mask_fn is None else valid & mask_fn(reference)
        rmse_reference, count = pixel_day_rmse(reference["LST"], observed, keep, day)
        rmse_perturbed, _ = pixel_day_rmse(perturbed, observed, keep, day)
        per_window.append(xr.Dataset({"rmse_reference": rmse_reference, "rmse_perturbed": rmse_perturbed, "n": count}))
    scores = xr.concat(per_window, dim="window").assign_coords(window=list(windows))
    scores["delta_rmse"] = scores["rmse_perturbed"] - scores["rmse_reference"]
    return scores


def pooled_rmse(rmse, n):
    """Count-weighted RMSE over the pixel dim."""
    return np.sqrt((rmse**2 * n).sum("pixel") / n.sum("pixel"))


def summary(scores):
    pooled_reference = pooled_rmse(scores["rmse_reference"], scores["n"])
    pooled_perturbed = pooled_rmse(scores["rmse_perturbed"], scores["n"])
    table = xr.Dataset(
        {
            "rmse_reference": pooled_reference,
            "rmse_perturbed": pooled_perturbed,
            "delta_rmse": pooled_perturbed - pooled_reference,
            "median_pixel_delta": scores["delta_rmse"].median("pixel"),
            "share_pixels_worse": (scores["delta_rmse"] > 0).mean("pixel"),
        }
    )
    return table.to_dataframe().reset_index()


def collect(paths, experiment, version, eval_name, out_dir, windows=WINDOWS):
    """Score every <modality>.<variable>.zarr in out_dir against the run's forecast.zarr."""
    forecast = xr.open_dataset(paths.forecast_path(experiment, version, eval_name))
    reference = prepare_initialised(forecast)
    observed = obs_cube(obs_series(forecast), reference)
    reference_amplitude = None
    if AMPLITUDE_WINDOW in windows:
        reference_amplitude = amplitude_errors(reference["LST"], observed, reference)
    stores = sorted(out_dir.glob("*.zarr"))
    per_variable = []
    for path in stores:
        perturbed = prepare_initialised(xr.open_dataset(path))["LST"]
        per_variable.append(score_cube(reference, observed, perturbed, windows, reference_amplitude))
    scores = xr.concat(per_variable, dim="variable").assign_coords(variable=[path.stem for path in stores])
    scores.to_netcdf(out_dir / "scores.nc")
    table = summary(scores)
    table.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    print(table[table["window"] == "all"].to_string(index=False))
    return scores
