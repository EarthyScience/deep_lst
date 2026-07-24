"""Climatology baseline: the forecast at valid_time t is the mean of all clear-sky training-year
observations that match t's day-of-year and hour-of-day. The smoothed mode averages over a
circular day-of-year window instead of the exact day."""

from pathlib import Path

import numpy as np

from deep_lst.baselines.base import _ANCHOR_MODALITY, BaselineExtractor


class ClimatologyExtractor(BaselineExtractor):
    def __init__(
        self, var_name: str, test_year: int, smooth: bool = False, window_days: int = 7, overwrite: bool = False
    ):
        super().__init__(var_name, test_year, overwrite)
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
