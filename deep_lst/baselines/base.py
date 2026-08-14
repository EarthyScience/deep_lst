"""Baseline extractor base class.

Baselines are reference forecasts evaluated against the model on the same eval pixels. Each
subclass acquires a variable time series at those pixels from the preloaded columns (the same
cache inference builds), then reduces it to a per-valid_time forecast.

Pixel coords come from the preload's cell_lonlat and the dense-grid geometry is reconstructed by
pixel_grid, so baselines share the forecast.zarr layout exactly (dense lat/lon for box regions,
sparse pixel for scattered training sites) — no sampled store involved.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata import get_data_reader
from eodata.grids import T1H

from deep_lst.dataset.dataset import PRELOAD_END_DAY, PRELOAD_START_DAY, STORE_EPOCH_OFFSET
from deep_lst.evaluation.spatial import pixel_grid
from deep_lst.utils import ExperimentPaths, StatsZarrNormalizer
from deep_lst.utils.io import stats_spatial

_SPARSE_FILL_THRESHOLD = 0.01
_ANCHOR_MODALITY = "seviri"


class BaselineExtractor(ABC):
    def __init__(self, var_name: str, test_year: int, overwrite: bool = False, force_sparse: bool = False):
        self.var_name = var_name
        self.test_year = test_year
        self.overwrite = overwrite
        self.force_sparse = force_sparse
        self.paths = ExperimentPaths()

    @abstractmethod
    def _acquire(self, shared: dict, config: dict) -> None:
        """Populate self.spatial_idx, self.timestamps, self.var (n_pixels, n_time), self.pixel_lonlat."""

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def output_path(self) -> Path:
        pass

    def run(self, shared: dict, config: dict, store_key: str) -> None:
        self.store_key = store_key
        self.config = config
        if self.output_path().exists() and not self.overwrite:
            print(f"Output already exists for '{store_key}'. Pass overwrite=True to refresh.")
            return
        self._acquire(shared, config)
        self._resolve_spatial_layout()
        self._compute()
        self._save()

    def _preload_series(self, shared: dict, modality: str, var_name: str):
        """Physical (denormalised) time series of one modality variable at the eval pixels.

        Returns (spatial_idx, timestamps, var, pixel_lonlat): var is (n_pixels, n_time) with NaN
        where the store had no valid observation; timestamps span the preloaded hour axis. The
        values come from `modality` (nearest-neighbour mapped for coarse modalities) but the coords
        are always the seviri eval-pixel locations, so every baseline shares the forecast.zarr grid.
        """
        names = shared["var_names"][modality]["data"]
        channel = names.index(var_name)
        inverse = shared["inverse"][modality]
        column = shared["columns"][modality]["data"][:, :, channel].numpy()  # (n_unique, n_hours), normalised

        normaliser = StatsZarrNormalizer(
            self.paths.stats_zarr_path(modality, stats_spatial(self.config["test_dataset"]))
        )
        var = normaliser.denormalise(column[inverse], var_name)  # (n_pixels, n_hours), physical, NaN kept

        n_hours = (PRELOAD_END_DAY - PRELOAD_START_DAY) * 24
        start = pd.Timestamp(T1H.index_to_startdate(int(STORE_EPOCH_OFFSET))).tz_localize(None)
        timestamps = pd.date_range(start=start, periods=n_hours, freq="h")

        anchor_inverse = shared["inverse"][_ANCHOR_MODALITY]
        pixel_lonlat = shared["cell_lonlat"][_ANCHOR_MODALITY][anchor_inverse]
        return shared["anchor_ids"], timestamps, var, pixel_lonlat

    def _metadata_attrs(self, modality: str, store_name: str, var_name: str) -> dict:
        """Baseline attrs sourced from the store's own per-variable metadata scheme."""
        reader = get_data_reader(modality, store=str(self.paths.eozarr_store_path(store_name)), average_hourly=False)
        variable_meta = reader.data_arr.attrs.get("metadata", {}).get(var_name, {})
        return dict(
            variable=var_name,
            units=variable_meta.get("units", ""),
            long_name=variable_meta.get("long_name", var_name),
            description=variable_meta.get("description", ""),
            test_year=str(self.test_year),
        )

    def _resolve_spatial_layout(self) -> None:
        self.n_pixels = len(self.spatial_idx)
        self.lats, self.lons, self.local_rows, self.local_cols, self.grid_shape = pixel_grid(
            self.pixel_lonlat[:, 0], self.pixel_lonlat[:, 1]
        )
        fill = self.n_pixels / (self.grid_shape[0] * self.grid_shape[1])
        # force_sparse must mirror the inference flag: baselines and forecast.zarr have to
        # share one layout, and baseline_cube refuses to mix them.
        self.sparse = self.force_sparse or fill < _SPARSE_FILL_THRESHOLD
        print(f"Spatial: {self.n_pixels} pixels, fill={fill:.4f} → {'sparse' if self.sparse else 'dense'} output")

    def _make_dataset(self, baseline: np.ndarray, test_timestamps: pd.DatetimeIndex, attrs: dict) -> xr.Dataset:
        # Coords are float64 BY CONTRACT: the forecast zarr writes float64 lat/lon, and
        # xarray aligns coordinates by exact value — a float32 baseline grid silently
        # intersects to empty against a float64 forecast grid.
        if self.sparse:
            pixel_lats = self.lats[self.local_rows].astype(np.float64)
            pixel_lons = self.lons[self.local_cols].astype(np.float64)
            return xr.Dataset(
                {self.var_name: (["valid_time", "pixel"], baseline)},
                coords={
                    "valid_time": test_timestamps.values,
                    "lat": ("pixel", pixel_lats),
                    "lon": ("pixel", pixel_lons),
                },
                attrs=attrs,
            )
        out = np.full((len(test_timestamps), *self.grid_shape), np.nan, dtype=np.float32)
        out[:, self.local_rows, self.local_cols] = baseline
        return xr.Dataset(
            {self.var_name: (["valid_time", "lat", "lon"], out)},
            coords={
                "valid_time": test_timestamps.values,
                "lat": self.lats.astype(np.float64),
                "lon": self.lons.astype(np.float64),
            },
            attrs=attrs,
        )

    def _save(self) -> None:
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


class ModalityBaselineExtractor(BaselineExtractor):
    """Cross-modality reference baseline: one modality variable at the eval pixels for the test year.

    The variable is read from the preload (so it must be in that modality's `load`), taken as-is for
    the test year — no temporal reduction. Subclasses only set `modality`; the variable is flexible
    (e.g. era5_land skt or swvl1). Coords stay on the seviri side, so the baseline aligns with
    forecast.zarr and the other baselines.
    """

    modality: str = None

    def _acquire(self, shared: dict, config: dict) -> None:
        self.spatial_idx, self.timestamps, self.var, self.pixel_lonlat = self._preload_series(
            shared, self.modality, self.var_name
        )
        self.store_name = config["test_dataset"][self.modality]["store"]

    def _compute(self):
        test_mask = self.timestamps.year == self.test_year
        baseline = self.var[:, test_mask].T.astype(np.float32)  # (n_time, n_pixels)
        test_timestamps = self.timestamps[test_mask]
        attrs = self._metadata_attrs(self.modality, self.store_name, self.var_name)
        self.result = self._make_dataset(baseline, test_timestamps, attrs)

    def output_path(self) -> Path:
        return self.paths.modality_baseline_path(self.modality, self.store_key)
