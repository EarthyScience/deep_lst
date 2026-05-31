"""
Baseclass for extracting benchmarks from zarr data bases.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deep_lst.evaluation.spatial import lookup_pixel_coords
from deep_lst.utils import ExperimentPaths

_SPARSE_FILL_THRESHOLD = 0.01


class BaselineExtractor(ABC):
    def __init__(
        self,
        var_name: str,
        test_year: int,
        overwrite: bool = False,
    ):
        self.var_name = var_name
        self.test_year = test_year
        self.overwrite = overwrite
        self.paths = ExperimentPaths()

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def output_path(self) -> Path:
        pass

    def run(self, store: str, modality: str):
        self.store_key = store
        store_path = self.paths.sampled_store_path(modality, self.store_key)
        self.zarr_store = zarr.open(str(store_path), mode="r")

        if self.output_path().exists() and not self.overwrite:
            print(f"Output already exists for store '{store}'. Pass overwrite=True to refresh.")
            return

        self._resolve_store()
        self._resolve_spatial_layout()
        self._compute()
        self._save()

    def _save(self):
        path = self.output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.result.to_netcdf(path)
        print(f"Saved: {path}")
        if "pixel" in self.result.dims:
            print(f"  sparse: {self.result.sizes['pixel']} pixels, {self.result.sizes['valid_time']} timesteps")
        else:
            print(
                f"  dense: {self.result.sizes['lat']}×{self.result.sizes['lon']}, {self.result.sizes['valid_time']} timesteps"
            )

    def _resolve_store(self):
        self.spatial_idx = self.zarr_store["spatial_idx"][:]
        self.timestamps = pd.DatetimeIndex(self.zarr_store["time_idx"][:])
        self.var = self.zarr_store[self.var_name][:]  # (n_pixels, n_time) float32
        self.n_pixels = self.var.shape[0]

    def _resolve_spatial_layout(self):
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = lookup_pixel_coords(
            self.zarr_store, self.spatial_idx
        )
        fill = self.n_pixels / (self.grid_shape[0] * self.grid_shape[1])
        self.sparse = fill < _SPARSE_FILL_THRESHOLD
        print(f"Spatial: {self.n_pixels} pixels, fill={fill:.4f} → {'sparse' if self.sparse else 'dense'} output")

    def _make_dataset(self, baseline: np.ndarray, test_timestamps: pd.DatetimeIndex, attrs: dict) -> xr.Dataset:
        if self.sparse:
            pixel_lats = self.lats[self.local_rows]
            pixel_lons = self.lons[self.local_cols]
            return xr.Dataset(
                {self.var_name: (["valid_time", "pixel"], baseline)},
                coords={
                    "valid_time": test_timestamps.values,
                    "lat": ("pixel", pixel_lats),
                    "lon": ("pixel", pixel_lons),
                },
                attrs=attrs,
            )
        else:
            out = np.full((len(test_timestamps), *self.grid_shape), np.nan, dtype=np.float32)
            out[:, self.local_rows, self.local_cols] = baseline
            return xr.Dataset(
                {self.var_name: (["valid_time", "lat", "lon"], out)},
                coords={"valid_time": test_timestamps.values, "lat": self.lats, "lon": self.lons},
                attrs=attrs,
            )


class PersistenceExtractor(BaselineExtractor):
    def _compute(self):
        # group by time of day.
        time_of_day = self.timestamps.time  # array of datetime.time objects, length n_time
        unique_times = np.unique(time_of_day)

        hour_wise_persistence = []
        for hour in unique_times:
            hourly_mask = time_of_day == hour  # boolean mask of length n_time.
            hour_timestamps = self.timestamps[hourly_mask]
            test_mask = hour_timestamps.year == self.test_year

            hour_lst = self.var[:, hourly_mask]  # (n_pixels, n_time_at_this_hour)

            df = pd.DataFrame(hour_lst.T, index=hour_timestamps)  # (n_time_at_this_hour, n_pixels)
            persistence = df.shift(
                1
            ).ffill()  # shift all values by one day. Then forward propagate last observed value.

            hour_wise_persistence.append(persistence.loc[test_mask])

        results_df = pd.concat(hour_wise_persistence).sort_index()
        persistence = results_df.values.astype(np.float32)  # remains as valid_time x pixel.
        test_timestamps = pd.DatetimeIndex(results_df.index)

        attrs = dict(
            test_year=str(self.test_year),
            variable=self.var_name,
            description=(
                "Persistence forecast for test year."
                "Persistence forecast at valid_time t uses the last clear sky obs from (t - 1) backward with now limit. "
                "run_evaluation.py applies the lead_time shift during alignment."
            ),
        )

        self.result = self._make_dataset(persistence, test_timestamps, attrs)

    def output_path(self) -> Path:
        return self.paths.persistence_path(self.store_key)


class ClimatologyExtractor(BaselineExtractor):
    def __init__(self, var_name: str, test_year: int, window_days: int = 7, overwrite: bool = False):
        super().__init__(var_name, test_year, overwrite)
        self.window_days = window_days

    def _compute(self):
        train_mask = self.timestamps.year < self.test_year
        test_mask = self.timestamps.year == self.test_year

        train_lst = self.var[:, train_mask].astype(np.float64)
        train_doys = self.timestamps[train_mask].day_of_year.values
        train_hours = self.timestamps[train_mask].hour.values
        test_ts = self.timestamps[test_mask]
        train_years = f"{self.timestamps[train_mask][0].year}–{self.timestamps[train_mask][-1].year}"

        if self.mode == "hourly":
            self._build_hourly_climatology(train_lst, train_doys, train_hours)
        else:
            self._build_smoothed_hourly_climatology(train_lst, train_doys, train_hours)

        baseline = self._lookup_climatology(test_ts)

        attrs_base = dict(
            test_year=str(self.test_year),
            train_years=train_years,
            units="K",
            description=(
                f"{self.mode.replace('_', ' ').capitalize()} climatology forecast for test year."
                "Climatology forecast at valid_time t uses the mean of all clear sky obs from the training years that match the same day of year and hour of day as t."
            ),
        )

        self.result = self._make_dataset(baseline, test_ts, attrs_base)

    def output_path(self) -> Path:
        return self.paths.climatology_path(self.store_key, self.mode)

    def run(self, store: str, modality: str, mode: str = "hourly"):
        self.mode = mode
        super().run(store, modality)

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
                # Build the circular DOY window
                offsets = np.arange(-self.window_days, self.window_days + 1)
                target_doys = ((doy - 1 + offsets) % 366) + 1  # 1-based, wrap
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


class Era5Extractor(BaselineExtractor):
    def __init__(self, var_name: str, test_year: int, overwrite: bool = False):
        super().__init__(var_name, test_year, overwrite)

    def _compute(self):
        test_mask = self.timestamps.year == self.test_year
        test_ts = self.timestamps[test_mask]
        baseline = self.var[:, test_mask].T  # (n_era5_pixels, n_test)

        attrs = dict(
            test_year=str(self.test_year),
            units="K",
            description="ERA5 skin temperature baseline at SEVIRI eval pixel locations.",
        )
        self.result = self._make_dataset(baseline, test_ts, attrs)

    def output_path(self) -> Path:
        return self.paths.baseline_path(self.store_key)

    def run(self, store: str):
        super().run(store, modality="era5")

    def _resolve_store(self):
        idx_path = self.paths.spatial_indices_path("era5", self.store_key)

        idx_data = np.load(idx_path)
        self.src_indices = idx_data["src_indices"]  # (n,) SEVIRI flat pixel indices
        self.mapping = idx_data["mapping"]  # (n,) ERA5 flat pixel indices, aligned 1-to-1

        dst_spatial_idx = self.zarr_store["spatial_idx"][:]
        self.timestamps = pd.DatetimeIndex(self.zarr_store["time_idx"][:])
        skt = self.zarr_store[self.var_name][:]  # (n_pixels, n_time) float32

        # --- Map ERA5 store ranks for each SEVIRI pixel ---
        era5_flat_to_rank = {int(v): i for i, v in enumerate(dst_spatial_idx)}
        seviri_to_era5_rank = np.array([era5_flat_to_rank[int(e)] for e in self.mapping])

        skt_mapped = skt[seviri_to_era5_rank, :]  # (n_seviri, n_test)

        # --- Remap to SEVIRI spatial layout ---
        self.spatial_idx = np.unique(self.src_indices)
        self.n_pixels = len(self.spatial_idx)

        # pixel_positions[i] = rank of self.src_indices in unique_seviri
        _, pixel_positions = np.unique(self.src_indices, return_inverse=True)

        # skt_pixels[i, :] = SKT for unique_seviri[i] — shape (n_unique, n_test)
        self.var = skt_mapped[pixel_positions, :]
