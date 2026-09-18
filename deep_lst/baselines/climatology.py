"""Climatology baseline: the forecast at valid_time t is the mean of all clear-sky training-year
observations that match t's day-of-year and hour-of-day. The smoothed mode averages over a
circular day-of-year window instead of the exact day."""

import warnings
from pathlib import Path

import numpy as np

from deep_lst.baselines.base import _ANCHOR_MODALITY, BaselineExtractor
from deep_lst.evaluation_climpred.masks import DAY_HOURS, NIGHT_HOURS

# Reference-period anomaly statistics written next to the climatology (see
# anomaly_statistics): empirical quantiles of the dekad means, the valid daily values a
# dekad mean needs, the valid pixels an area-mean day needs.
QUANTILES = (0.05, 0.5, 0.95, 0.98)
MIN_DEKAD_DAYS = 2
MIN_PIXEL_FRACTION = 0.0  # no coverage guard on the area mean for now (any valid pixel counts)


def dekad_means(daily, day_dates, min_days=MIN_DEKAD_DAYS):
    """Calendar-dekad means (days 1-10, 11-20, 21-end) of `daily` (n_days, ...) along the
    first axis; NaN where a dekad has fewer than min_days valid values.
    Returns (means (n_dekads, ...), the calendar month of each dekad)."""
    dekad_index = np.minimum((day_dates.day.values - 1) // 10, 2)
    group = day_dates.year.values * 300 + day_dates.month.values * 3 + dekad_index
    groups, inverse = np.unique(group, return_inverse=True)
    means = np.full((len(groups), *daily.shape[1:]), np.nan, dtype=np.float32)
    months = np.zeros(len(groups), dtype=int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN dekads
        for position in range(len(groups)):
            rows = daily[inverse == position]
            valid = np.isfinite(rows).sum(axis=0)
            mean = np.nanmean(rows, axis=0)
            means[position] = np.where(valid >= min_days, mean, np.nan)
            months[position] = (groups[position] // 3) % 100
    return means, months


def _skewness(rows):
    """Fisher-Pearson skewness over the first axis, NaN-aware."""
    centred = rows - np.nanmean(rows, axis=0)
    second = np.nanmean(centred**2, axis=0)
    third = np.nanmean(centred**3, axis=0)
    return third / second**1.5


def anomaly_statistics(daily_anom, day_dates):
    """Reference-period statistics of daily-mean anomalies (n_days, n_pixels), per calendar
    month: per pixel the std and skewness of the daily values and the std and empirical
    QUANTILES of the calendar-dekad means; for the area mean over the pixels the std of the
    daily and of the dekad values. All stds ddof=1. A quantity shown at one of these
    aggregations must be standardized by the spread of the same aggregation, so the
    ±1 / ±2 sigma classes keep their meaning.
    Returns {name: (dims, array)}; names are the suffixes after '<var>_anom_'."""
    months = day_dates.month.values
    n_pixels = daily_anom.shape[1]
    dekad, dekad_months = dekad_means(daily_anom, day_dates)
    valid_fraction = np.isfinite(daily_anom).mean(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        area_daily = np.where(valid_fraction >= MIN_PIXEL_FRACTION, np.nanmean(daily_anom, axis=1), np.nan)
    area_dekad, _ = dekad_means(area_daily[:, None], day_dates)

    daily_std = np.full((12, n_pixels), np.nan, dtype=np.float32)
    daily_skew = np.full((12, n_pixels), np.nan, dtype=np.float32)
    dekad_std = np.full((12, n_pixels), np.nan, dtype=np.float32)
    dekad_quantile = np.full((len(QUANTILES), 12, n_pixels), np.nan, dtype=np.float32)
    daily_std_area = np.full(12, np.nan, dtype=np.float32)
    dekad_std_area = np.full(12, np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels, single-sample stds
        for month in range(1, 13):
            in_month = months == month
            in_dekad_month = dekad_months == month
            daily_std[month - 1] = np.nanstd(daily_anom[in_month], axis=0, ddof=1)
            daily_skew[month - 1] = _skewness(daily_anom[in_month])
            dekad_std[month - 1] = np.nanstd(dekad[in_dekad_month], axis=0, ddof=1)
            dekad_quantile[:, month - 1] = np.nanquantile(dekad[in_dekad_month], QUANTILES, axis=0)
            daily_std_area[month - 1] = np.nanstd(area_daily[in_month], ddof=1)
            dekad_std_area[month - 1] = np.nanstd(area_dekad[in_dekad_month, 0], ddof=1)
    return {
        "daily_std": (("month", "pixel"), daily_std),
        "daily_skew": (("month", "pixel"), daily_skew),
        "dekad_std": (("month", "pixel"), dekad_std),
        "dekad_quantile": (("quantile", "month", "pixel"), dekad_quantile),
        "daily_std_area": (("month",), daily_std_area),
        "dekad_std_area": (("month",), dekad_std_area),
    }


class ClimatologyExtractor(BaselineExtractor):
    def __init__(
        self,
        var_name: str,
        test_year: int,
        smooth: bool = False,
        window_days: int = 7,
        overwrite: bool = False,
        force_sparse: bool = False,
    ):
        super().__init__(var_name, test_year, overwrite, force_sparse)
        self.smooth = smooth
        self.window_days = window_days
        self.mode = "smoothed_hourly" if smooth else "hourly"

    def _acquire(self, shared: dict, config: dict) -> None:
        self.spatial_idx, self.timestamps, self.var, self.pixel_lonlat = self._preload_series(
            shared, _ANCHOR_MODALITY, self.var_name
        )

    def _compute(self):
        train_mask = self.timestamps.year < self.test_year
        test_mask = self.timestamps.year == self.test_year

        train_lst = self.var[:, train_mask].astype(np.float64)
        train_doys = self.timestamps[train_mask].day_of_year.values
        train_hours = self.timestamps[train_mask].hour.values
        test_ts = self.timestamps[test_mask]
        train_years = f"{self.timestamps[train_mask][0].year}–{self.timestamps[train_mask][-1].year}"

        if self.smooth:
            self._build_smoothed_hourly_climatology(train_lst, train_doys, train_hours)
        else:
            self._build_hourly_climatology(train_lst, train_doys, train_hours)

        baseline = self._lookup_climatology(test_ts)

        attrs = dict(
            test_year=str(self.test_year),
            train_years=train_years,
            variable=self.var_name,
            units="K",
            description=(
                f"{self.mode.replace('_', ' ').capitalize()} climatology forecast for test year. "
                "Climatology at valid_time t is the mean of all clear-sky training-year obs matching t's "
                "day of year and hour of day."
            ),
        )
        self.result = self._make_dataset(baseline, test_ts, attrs)
        monthly_stats = self._monthly_anomaly_stats(train_lst, self.timestamps[train_mask])
        self._attach_monthly_stats(monthly_stats)

    def output_path(self) -> Path:
        return self.paths.climatology_path(self.store_key, self.mode)

    def _build_hourly_climatology(self, train_lst, doys, hours):
        self.clim = np.full((367, 24, self.n_pixels), np.nan, dtype=np.float64)
        for doy in np.unique(doys):
            for hour in range(24):
                mask = (doys == doy) & (hours == hour)
                if mask.sum() == 0:
                    continue
                self.clim[doy, hour, :] = np.nanmean(train_lst[:, mask], axis=1)

    def _build_smoothed_hourly_climatology(self, train_lst, doys, hours):
        self.clim = np.full((367, 24, self.n_pixels), np.nan, dtype=np.float64)
        for hour in range(24):
            hour_mask = hours == hour
            if hour_mask.sum() == 0:
                continue
            lst_hour = train_lst[:, hour_mask]  # (n_pixels, n_h)
            doys_h = doys[hour_mask]  # (n_h,)
            for doy in range(1, 366 + 1):
                offsets = np.arange(-self.window_days, self.window_days + 1)
                target_doys = ((doy - 1 + offsets) % 366) + 1  # 1-based, circular
                in_window = np.isin(doys_h, target_doys)
                if in_window.sum() == 0:
                    continue
                self.clim[doy, hour, :] = np.nanmean(lst_hour[:, in_window], axis=1)

    def _lookup_climatology(self, test_timestamps):
        result = np.full((len(test_timestamps), self.n_pixels), np.nan, dtype=np.float32)
        doys = test_timestamps.day_of_year.values
        hours = test_timestamps.hour.values
        for i, (doy, hour) in enumerate(zip(doys, hours)):
            result[i] = self.clim[doy, hour, :]
        return result

    def _monthly_anomaly_stats(self, train_lst, train_ts):
        """Reference-period statistics of the daily-mean anomalies vs this climatology
        (anomaly_statistics), per calendar month: all hours, plus the solar day/night
        windows of the verification masks. Night hours are binned onto the calendar day
        they fall on. Variable names: <var>_anom_<statistic><window suffix>."""
        train_clim = self._lookup_climatology(train_ts)
        hourly_anom = train_lst.T.astype(np.float32) - train_clim
        assert train_ts[0].hour == 0 and len(train_ts) % 24 == 0
        n_days = len(train_ts) // 24
        day_dates = train_ts[::24]

        lons = self.pixel_lonlat[:, 0]
        solar = (train_ts.hour.values[:, None] + lons[None, :] / 15.0) % 24.0
        hour_masks = {
            "": None,
            "_day": (solar >= DAY_HOURS[0]) & (solar <= DAY_HOURS[1]),
            "_night": (solar >= NIGHT_HOURS[0]) | (solar <= NIGHT_HOURS[1]),
        }
        del solar

        stats = {}
        for suffix, hour_mask in hour_masks.items():
            masked = hourly_anom if hour_mask is None else np.where(hour_mask, hourly_anom, np.float32(np.nan))
            daily_anom = np.nanmean(masked.reshape(n_days, 24, self.n_pixels), axis=1)
            for statistic, (dims, values) in anomaly_statistics(daily_anom, day_dates).items():
                stats[f"{self.var_name}_anom_{statistic}{suffix}"] = (dims, values)
        return stats

    def _attach_monthly_stats(self, stats):
        for name, (dims, values) in stats.items():
            if "pixel" in dims and not self.sparse:
                dense = np.full((*values.shape[:-1], *self.grid_shape), np.nan, dtype=np.float32)
                dense[..., self.local_rows, self.local_cols] = values
                self.result[name] = (dims[:-1] + ("lat", "lon"), dense)
            else:
                self.result[name] = (dims, values)
        self.result = self.result.assign_coords(month=np.arange(1, 13), quantile=np.array(QUANTILES))
