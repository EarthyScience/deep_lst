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
import xarray as xr

from .common import baseline_files, lookback_hours, nearest_pixel
from .cubes import obs_series, prepare_initialised, series_at_pixel
from .plots import SITE_STYLES, plot_site
from .stats import select_init


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
