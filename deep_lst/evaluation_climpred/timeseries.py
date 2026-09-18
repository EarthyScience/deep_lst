"""Site time series: the one figure built from the forecast store rather than scores/.

One init per season at each named site, drawn as prediction / climatology /
observations over the whole horizon. The aggregate scores say the model beats
climatology; this shows what that claim looks like on a single forecast, where error
growth, the horizon and the gap to the climatology are all visible at once.

Kept apart from evaluate.py because it is the only product needing the forecast zarr
and the baseline series -- it carries the whole memory cost of the evaluation run
(~4.5 GB, ~1:45) while everything else reads the small derived files.
"""

import numpy as np
import pandas as pd
import xarray as xr

from .common import baseline_files, lookback_hours, nearest_pixel, verification_dir
from .cubes import attach_coverage, obs_series, prepare_initialised, series_at_pixel
from .plots import REGIME_COLUMNS, SITE_STYLES, _day_positions, plot_regime_days, plot_site
from .stats import select_clear_lookback_init, select_init, select_regime_pixels

EXAMPLE_INIT_HOUR = 0  # UTC: every drawn forecast day then spans the valid hours 01-24


def plot_site_forecasts(args, plots_dir):
    spatial = args.paths.baseline_spatial(args.eval_name)
    forecast = xr.open_dataset(args.paths.forecast_path(args.experiment, args.version, args.eval_name))
    initilised = prepare_initialised(forecast)
    obs = obs_series(forecast)
    climatology_series = xr.open_dataset(baseline_files(args.paths, spatial)["clim_smooth"])

    lookback = lookback_hours(args.paths, args.experiment, args.version)
    init_times = initilised["init"].values
    valid_times = init_times[:, None] + initilised["lead"].values[None, :] * np.timedelta64(1, "h")
    lookback_times = init_times[:, None] + np.arange(-lookback, 0)[None, :] * np.timedelta64(1, "h")

    for site in SITE_STYLES:
        pixel = nearest_pixel(initilised, site, max_offset_deg=0.1)
        if pixel is None:
            continue
        pred = initilised["LST"].isel(pixel=pixel).values
        observed = series_at_pixel(obs, valid_times, pixel)
        climatology = series_at_pixel(climatology_series, valid_times, pixel)
        lookback_obs = series_at_pixel(obs, lookback_times, pixel)
        chosen = select_init(pred, observed, climatology, init_times)
        plot_site(
            site,
            chosen,
            pred,
            observed,
            climatology,
            lookback_obs,
            initilised["lead"].values,
            lookback,
            init_times,
            plots_dir,
        )


def plot_regime_forecasts(args, plots_dir, days=(1, 10)):
    """One example pixel per climate regime, drawn at the forecast `days`.

    The pixel comes from the per-pixel scores (upper quartile of the last drawn
    day's skill against the climatology within its regime); the init is the one with
    the fullest lookback window at that pixel among the EXAMPLE_INIT_HOUR inits whose
    drawn days are completely forecast and observed. Only the chosen pixels are read
    from the store, so this stays cheap next to the site figure."""
    spatial = args.paths.baseline_spatial(args.eval_name)
    forecast = xr.open_dataset(args.paths.forecast_path(args.experiment, args.version, args.eval_name))
    pixel_daily = xr.open_dataset(verification_dir(args, "scores") / "pixel_daily.nc")
    pixels = select_regime_pixels(pixel_daily, day=days[-1])

    columns = [pixels[regime] for regime in REGIME_COLUMNS]
    subset = forecast.isel(pixel=columns)
    initilised = prepare_initialised(subset)
    obs = obs_series(subset)
    lookback = lookback_hours(args.paths, args.experiment, args.version)
    initilised = attach_coverage(initilised, obs, lookback)

    files = baseline_files(args.paths, spatial)
    climatology_series = xr.open_dataset(files["clim_smooth"])
    persistence_series = xr.open_dataset(files["persistence"])

    lead = initilised["lead"].values
    init_times = initilised["init"].values
    valid_times = init_times[:, None] + lead[None, :] * np.timedelta64(1, "h")
    day_positions = [_day_positions(lead, day) for day in days]
    eligible = pd.DatetimeIndex(init_times).hour == EXAMPLE_INIT_HOUR

    picks = {}
    for position, regime in enumerate(REGIME_COLUMNS):
        pixel = pixels[regime]
        pred = initilised["LST"].isel(pixel=position).values
        observed = series_at_pixel(obs, valid_times, position)
        climatology = series_at_pixel(climatology_series, valid_times, pixel)
        persistence = series_at_pixel(persistence_series, valid_times, pixel)
        coverage = initilised["lookback_coverage"].isel(pixel=position).values
        try:
            index = select_clear_lookback_init(coverage, pred, observed, day_positions, eligible)
        except ValueError as error:
            # An example figure must not fail the evaluation: the sparse validation
            # stores can leave a pixel without any usable init.
            print(f"Skipping the regime example figure: {regime} pixel {pixel} -- {error}")
            return
        picks[regime] = {
            "lead": lead,
            "valid_times": valid_times[index],
            "pred": pred[index],
            "observed": observed[index],
            "climatology": climatology[index],
            "persistence": persistence[index],
            "lon": float(initilised["lon"].isel(pixel=position)),
            "lat": float(initilised["lat"].isel(pixel=position)),
            "init": init_times[index],
            "coverage": float(coverage[index]),
        }
        print(f"regime example {regime}: pixel {pixel}, init {init_times[index]}, lookback {coverage[index]:.0%} clear")
    plot_regime_days(picks, days, plots_dir)
