"""RMSE change per pixel and forecast day when one input is perturbed, against the run's own
forecast.zarr, on the samples where both forecasts and the observation are valid.

Writes scores.nc (variable, window, day, pixel: rmse_reference, rmse_perturbed, delta_rmse, n)
and summary.csv (per variable, window, day: pooled RMSE of both, pooled delta, median pixel
delta, share of pixels with a positive delta) into the sensitivity folder.
"""

import numpy as np
import xarray as xr

from deep_lst.evaluation_climpred.cubes import obs_cube, obs_series, prepare_initialised
from deep_lst.evaluation_climpred.masks import MASKS

WINDOWS = ("all", "day", "night")


def pixel_day_rmse(cube, observed, keep, day):
    """(day, pixel) RMSE of one forecast cube over the kept (init, lead) samples."""
    squared = ((cube - observed) ** 2).where(keep)
    total = squared.groupby(day).sum("lead").sum("init")
    count = keep.groupby(day).sum("lead").sum("init")
    return np.sqrt(total / count), count


def score_store(reference, observed, perturbed_path, windows):
    """Per-window (day, pixel) RMSE of reference and perturbed forecasts on their joint samples."""
    perturbed = prepare_initialised(xr.open_dataset(perturbed_path))["LST"]
    valid = reference["LST"].notnull() & observed.notnull() & perturbed.notnull()
    day = xr.DataArray((reference["lead"].values - 1) // 24 + 1, dims="lead", name="day")
    per_window = []
    for window in windows:
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
    stores = sorted(out_dir.glob("*.zarr"))
    per_variable = [score_store(reference, observed, path, windows) for path in stores]
    scores = xr.concat(per_variable, dim="variable").assign_coords(variable=[path.stem for path in stores])
    scores.to_netcdf(out_dir / "scores.nc")
    table = summary(scores)
    table.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    print(table[table["window"] == "all"].to_string(index=False))
    return scores
