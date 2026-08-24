"""Standardized daytime LST anomaly index for the Zambia 2024 drought case study.

Builds the index the case-study figures are drawn from, for observations and for three
forecast views (12 h lead, 9.5 d lead, seasonal trajectories), and writes the figures.

Definition of the index
-----------------------
1. Hourly anomaly. At every (pixel, hour) of the test year, the anomaly is the
   observation minus the SMOOTHED hourly climatology at the same (pixel, day-of-year,
   hour-of-day). The climatology is the mean of all clear-sky training-year
   observations whose day-of-year falls in a circular +/- SMOOTHING_WINDOW_DAYS window
   around the target day, taken separately per hour-of-day
   (deep_lst/baselines/climatology.py, ClimatologyExtractor with smooth=True). Forecast
   anomalies use the same climatology, subtracted at the forecast's valid time.

2. Daytime window. Only hours whose LOCAL SOLAR hour falls in DAY_HOURS (inclusive at
   both ends) are kept. Local solar hour is (UTC hour + lon / 15) mod 24. DAY_HOURS is
   imported from deep_lst.evaluation_climpred.masks -- the single definition shared with
   the verification masks and with the sigma below, so the three can never disagree.

3. Daily mean. The kept hourly anomalies are averaged per calendar day and pixel, and
   the day is kept only when at least MIN_DAY_HOURS of them are present. Cloud gaps make
   this bite for observations; model output has no gaps, so it is inert for the
   seasonal-trajectory series (verified: identical series at 2 and at 3).
   EXCEPTION: the fixed-lead-band series (12 h, 9.5 d) average the in-window leads of one
   forecast day with NO minimum count, matching the published figure; the difference
   against MIN_DAY_HOURS is at most 0.002 sigma on the area mean.

4. Standardization. Each daily anomaly is divided by the per-pixel, per-calendar-month
   standard deviation (ddof=1) of daily anomalies over the DAY_HOURS window, computed
   from TRAINING YEARS ONLY (2016-2023; the test year is excluded by construction).
   That sigma ships with the climatology file as LST_anom_daily_std_day.

5. No distribution fit. Unlike SPI or SPAI, nothing is fitted and nothing is transformed
   to a normal deviate: the index is a plain z-score of the daily anomaly against a
   monthly climatological spread. Values are therefore NOT probabilities and their tails
   are not calibrated.

The maps show dekadal (day 1-10) means of the index; the time series show its spatial
mean over the region, raw and as a ROLLING_DAYS-day centred rolling mean (display only,
not part of the index).

    uv run scripts/analysis/zambia_anomaly_index.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from deep_lst.evaluation_climpred.cubes import baseline_cube, obs_series, prepare_initialised
from deep_lst.evaluation_climpred.masks import DAY_HOURS
from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()

REGION = "zambia_drought"
SEASONAL_REGION = "zambia_drought_seasonal"

MIN_DAY_HOURS = 3  # in-window hours a day needs to enter the index
SMOOTHING_WINDOW_DAYS = 7  # +/- window of the smoothed climatology (run_baselines --window_days)
ROLLING_DAYS = 10  # centred rolling mean, display only
# The training years and the test year are NOT set here: they are baked into the
# climatology file by run_baselines.py and printed back from its attrs.

SHORT_LEADS = slice(1, 24)  # first forecast day of a 00 UTC init -> "12h lead"
MEDIUM_LEADS = slice(217, 240)  # tenth forecast day -> "9.5d lead", plotted at init + 9 d
MEDIUM_SHIFT_DAYS = 9

DEKAD_MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04"]
DEKAD_TITLES = ["Jan 1–10 24", "Feb 1–10 24", "Mar 1–10 24", "Apr 1–10 24"]
SEASON = slice("2024-01-01", "2024-04-30")
SEASONAL_COLORS = {"2024-01-01": "tab:blue", "2024-02-01": "tab:orange", "2024-03-01": "tab:purple"}


def print_constants(clim):
    """The numbers the manuscript has to quote, read back from the data where possible."""
    print("--- standardized anomaly index constants ---")
    print(f"daytime window (local solar h, inclusive) : {DAY_HOURS[0]}-{DAY_HOURS[1]}")
    print(f"min in-window hours per day (N)           : {MIN_DAY_HOURS}")
    print(
        f"climatology smoothing window (days)       : +/-{SMOOTHING_WINDOW_DAYS} ({2 * SMOOTHING_WINDOW_DAYS + 1} d circular)"
    )
    print(f"climatology train years (file attr)       : {clim.attrs['train_years']}")
    print(f"climatology test year (file attr)         : {clim.attrs['test_year']}")
    print("sigma variable                            : LST_anom_daily_std_day (month, pixel), ddof=1")
    print(f"rolling mean for display (days)           : {ROLLING_DAYS}, centred, min_periods=5")
    print("distribution fit                          : none (plain z-score, not SPI/SPAI)")


def solar_hour(hours, lon):
    return (hours + lon / 15) % 24


def in_day_window(hours, lon):
    hour = solar_hour(hours, lon)
    return (hour >= DAY_HOURS[0]) & (hour <= DAY_HOURS[1])


def standardize(daily_anomaly, sigma):
    """Daily anomaly / the pixel's sigma for that calendar month."""
    return daily_anomaly / sigma.sel(month=daily_anomaly["time"].dt.month)


def daily_index(hourly_anomaly, sigma):
    """Hourly (time, pixel) anomalies -> daytime daily means in K and in sigma."""
    windowed = hourly_anomaly.where(in_day_window(hourly_anomaly["time"].dt.hour, hourly_anomaly["lon"]))
    mean = windowed.resample(time="1D").mean()
    count = windowed.resample(time="1D").count()
    daily = mean.where(count >= MIN_DAY_HOURS)
    return daily, standardize(daily, sigma)


def observed_index(forecast, climatology, sigma):
    observed = obs_series(forecast)["LST"]
    hourly_anomaly = observed - climatology["LST"].rename({"valid_time": "time"})
    return daily_index(hourly_anomaly, sigma)


def lead_band_index(initialised, clim_path, sigma, leads, shift_days=0):
    """Daytime mean of one fixed lead band per 00 UTC init, in K and in sigma.

    No minimum count here -- see point 3 of the module docstring."""
    band = initialised.sel(init=initialised["init"].dt.hour == 0).sel(lead=leads)
    anomaly = band["LST"] - baseline_cube(clim_path, band)
    valid_hour = (band["init"].dt.hour + band["lead"]) % 24
    daily = anomaly.where(in_day_window(valid_hour, band["lon"])).mean("lead").rename({"init": "time"})
    if shift_days:
        daily = daily.assign_coords(time=daily["time"] + np.timedelta64(shift_days, "D"))
    return daily, standardize(daily, sigma)


def seasonal_indices(seasonal_forecast, clim_path, sigma):
    """One index series per seasonal init, keyed by the init date."""
    initialised = prepare_initialised(seasonal_forecast)
    anomaly = initialised["LST"] - baseline_cube(clim_path, initialised)
    series = {}
    for position in range(initialised.sizes["init"]):
        trajectory = anomaly.isel(init=position)
        valid_times = trajectory["init"].values + trajectory["lead"].values * np.timedelta64(1, "h")
        trajectory = trajectory.rename({"lead": "time"}).assign_coords(time=valid_times).drop_vars("init")
        _, standardized = daily_index(trajectory, sigma)
        series[str(initialised["init"].values[position])[:10]] = standardized.mean("pixel")
    return series


def rolling(series):
    return series.rolling(time=ROLLING_DAYS, center=True, min_periods=5).mean()


def plot_observed(standardized, out_dir):
    """Four dekadal maps of the index over the region, plus its area-mean series."""
    grid = standardized.set_index(pixel=["lat", "lon"]).unstack("pixel")
    layout = [["jan", "feb", "mar", "apr"], ["series", "series", "series", "series"]]
    fig, axes = plt.subplot_mosaic(
        layout, figsize=(15, 7.5), constrained_layout=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )
    map_axes = [axes[key] for key in ("jan", "feb", "mar", "apr")]
    for ax, month, title in zip(map_axes, DEKAD_MONTHS, DEKAD_TITLES):
        field = grid.sel(time=slice(f"{month}-01", f"{month}-10")).mean("time")
        mesh = ax.pcolormesh(grid["lon"], grid["lat"], field, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_facecolor("0.85")
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Lon")
    map_axes[0].set_ylabel("Lat")
    for ax in map_axes[1:]:
        ax.set_yticklabels([])
    fig.colorbar(mesh, ax=map_axes, shrink=0.85, label="Standardized anomaly")

    series = standardized.mean("pixel")
    smooth = rolling(series)
    ax = axes["series"]
    for month in DEKAD_MONTHS:
        ax.axvspan(np.datetime64(f"{month}-01"), np.datetime64(f"{month}-10"), color="0.92", zorder=0)
    for level in (-1, 1):
        ax.axhline(level, color="0.8", lw=0.9, ls="--", zorder=1)
    for level in (-2, 2):
        ax.axhline(level, color="red", lw=0.9, ls="--", zorder=1)
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    ax.plot(series.sel(time=SEASON)["time"], series.sel(time=SEASON), color="0.6", lw=0.7, label="Daily mean")
    ax.plot(
        smooth.sel(time=SEASON)["time"],
        smooth.sel(time=SEASON),
        color="k",
        lw=1.8,
        label=f"{ROLLING_DAYS}-day rolling mean",
    )
    ax.set_xlim(np.datetime64("2024-01-01"), np.datetime64("2024-05-01"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Standardized anomaly", fontweight="bold")
    ax.set_xlabel("Time", fontweight="bold")
    ax.legend(loc="upper left", fontsize=16)
    fig.savefig(out_dir / "obs_daily_lst_anomalies.pdf")
    plt.close(fig)


def plot_lead_bands(observed_area, short_area, medium_area, out_dir):
    """Area-mean index, observed against the 12 h and 9.5 d forecast bands, in K and in sigma.

    Each argument is a (kelvin, sigma) pair of area-mean series."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    panels = [
        (axes[0], observed_area[0], short_area[0], medium_area[0], "Anomaly (K)"),
        (axes[1], observed_area[1], short_area[1], medium_area[1], "Standardized anomaly (σ)"),
    ]
    for ax, observed, short, medium, label in panels:
        ax.axhline(0, color="0.6", lw=0.8, zorder=1)
        entries = [
            (observed, "0.6", 1.0, "k", "Observed"),
            (short, "tab:blue", 0.5, "tab:blue", "12h lead"),
            (medium, "tab:orange", 0.5, "tab:orange", "9.5d lead"),
        ]
        for series, raw_color, raw_alpha, smooth_color, name in entries:
            raw = series.sel(time=SEASON)
            smooth = rolling(series).sel(time=SEASON)
            ax.plot(raw["time"], raw, color=raw_color, lw=0.7, alpha=raw_alpha)
            ax.plot(smooth["time"], smooth, color=smooth_color, lw=1.8, label=name)
        ax.set_ylabel(label)
    for level in (-2, -1, 1, 2):
        axes[1].axhline(level, color="0.8", lw=0.8, ls="--", zorder=1)
    axes[0].legend(frameon=False, fontsize=16, loc="upper left")
    axes[1].set_xlim(np.datetime64("2024-01-01"), np.datetime64("2024-05-01"))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.align_ylabels()
    fig.savefig(out_dir / "fc_daily_lst_anomalies.pdf")
    plt.close(fig)


def plot_seasonal(observed_sigma, trajectories, out_dir):
    """Observed index against the three seasonal forecast trajectories."""
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    for level in (-2, -1, 1, 2):
        ax.axhline(level, color="0.8", lw=0.9, ls="--", zorder=1)
    raw = observed_sigma.sel(time=SEASON)
    smooth = rolling(observed_sigma).sel(time=SEASON)
    ax.plot(raw["time"], raw, color="0.6", lw=0.7)
    ax.plot(smooth["time"], smooth, color="k", lw=1.8, label="Observed")
    for init_label, series in trajectories.items():
        color = SEASONAL_COLORS[init_label]
        ax.axvline(np.datetime64(init_label), color=color, lw=1.4, ls=":", zorder=1)
        ax.plot(series.sel(time=SEASON)["time"], series.sel(time=SEASON), color=color, lw=0.7, alpha=0.4)
        trajectory_smooth = rolling(series).sel(time=SEASON)
        ax.plot(trajectory_smooth["time"], trajectory_smooth, color=color, lw=1.8, label=f"Init {init_label[5:]}")
    ax.set_ylabel("Standardized anomaly", fontweight="bold", fontsize=16)
    ax.set_xlabel("Valid time", fontweight="bold", fontsize=16)
    ax.set_xlim(np.datetime64("2024-01-01"), np.datetime64("2024-05-01"))
    ax.legend(frameon=False, ncols=2, fontsize=16)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)
    fig.savefig(out_dir / "fc_daily_lst_anomalies_seasonal.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, default="mlp_smoothl1_h24_full_v13")
    parser.add_argument("--version", type=str, default="version_0")
    args = parser.parse_args()

    spatial = PATHS.baseline_spatial(REGION)
    clim_path = PATHS.climatology_path(spatial, "smoothed_hourly")
    climatology = xr.open_dataset(clim_path)
    sigma = climatology["LST_anom_daily_std_day"]
    print_constants(climatology)

    plt.rcParams.update({"axes.labelsize": 16, "axes.titlesize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16})

    forecast = xr.open_zarr(PATHS.forecast_path(args.experiment, args.version, REGION), decode_timedelta=True)
    out_dir = PATHS.results_plots_dir(args.experiment, args.version, REGION)

    observed_kelvin, observed_sigma = observed_index(forecast, climatology, sigma)
    plot_observed(observed_sigma, out_dir)

    initialised = prepare_initialised(forecast)
    short = lead_band_index(initialised, clim_path, sigma, SHORT_LEADS)
    medium = lead_band_index(initialised, clim_path, sigma, MEDIUM_LEADS, MEDIUM_SHIFT_DAYS)
    plot_lead_bands(
        (observed_kelvin.mean("pixel"), observed_sigma.mean("pixel")),
        (short[0].mean("pixel"), short[1].mean("pixel")),
        (medium[0].mean("pixel"), medium[1].mean("pixel")),
        out_dir,
    )

    seasonal = xr.open_zarr(PATHS.forecast_path(args.experiment, args.version, SEASONAL_REGION), decode_timedelta=True)
    trajectories = seasonal_indices(seasonal, clim_path, sigma)
    plot_seasonal(observed_sigma.mean("pixel"), trajectories, out_dir)

    print(
        f"wrote obs_daily_lst_anomalies.pdf, fc_daily_lst_anomalies.pdf, "
        f"fc_daily_lst_anomalies_seasonal.pdf to {out_dir}"
    )


if __name__ == "__main__":
    main()
