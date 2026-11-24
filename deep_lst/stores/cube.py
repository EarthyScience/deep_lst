"""Reader of a product store: one struct array data[time, lat, lon] per store.

Layout of such a store (see also tests/conftest.py, which writes small ones):

    data      (time, lat, lon)  one struct field per variable, sharded along time
    lat, lon  coordinates of the cell centres
    _indices  one row (0, time index) per written time step

The time dimension spans the whole time axis, so a time index is a position on T1H or T1D. A
static store has a single time step. Values are stored as written by the producer, often as
scaled integers; a reader for a concrete store is a subclass that only sets class attributes.
"""

from pathlib import Path

import numpy as np
import zarr

from deep_lst.stores.base import StoreReader
from deep_lst.stores.grid import LatLonGrid


class CubeReader(StoreReader):
    """Reads a (time, lat, lon) product store with plain zarr.

    Subclasses set:
        MODALITY_ID: The modality_id in the root attributes of the store.
        TIME_GRID: T1H or T1D; None for a static store.
        SCALE: {variable: factor}; the stored value is multiplied by it. Default 1.
        OFFSET: {variable: offset}; added after scaling. Default 0.

    Attributes:
        store: Folder of the store.
        data_arr: The zarr array data[time, lat, lon].
    """

    MODALITY_ID = None
    TIME_GRID = None
    SCALE = {}
    OFFSET = {}

    def __init__(self, store):
        self.store = Path(store)
        self.modality_id = self.MODALITY_ID
        self.time_grid = self.TIME_GRID
        self._check_modality()
        self.data_arr = zarr.open_array(str(self.store / "data"), mode="r")
        self.variables = list(self.data_arr.dtype.names)
        lat = zarr.open_array(str(self.store / "lat"), mode="r")[:]
        lon = zarr.open_array(str(self.store / "lon"), mode="r")[:]
        self.spatial_grid = LatLonGrid(lat, lon, name=self.modality_id)

    def _check_modality(self):
        """A reader decodes one kind of store: refuse a store of another modality."""
        attributes = zarr.open_group(str(self.store), mode="r").attrs
        stored = attributes["modality_id"]
        if stored != self.MODALITY_ID:
            name = type(self).__name__
            raise ValueError(f"{name} reads {self.MODALITY_ID!r}, but {self.store} holds {stored!r}")

    @property
    def available_indices(self):
        """(0, time index) of every written time step. (n, 2)

        The spatial index 0 stands for the whole grid: a time step is written for all cells.
        """
        return zarr.open_array(str(self.store / "_indices"), mode="r")[:]

    def decode(self, stored):
        """Stored struct values -> physical values.

        Args:
            stored: Struct array of any shape, as read from data_arr.

        Returns:
            float32 array of shape stored.shape + (n_variables,). NaN where the store holds the
            fill value of a variable.
        """
        decoded = np.empty(stored.shape + (len(self.variables),), dtype=np.float32)
        for channel, variable in enumerate(self.variables):
            raw = stored[variable]
            fill_value = self.data_arr.fill_value[variable]
            missing = raw == fill_value
            if raw.dtype.kind == "f":
                missing = missing | np.isnan(raw)
            scale = np.float32(self.SCALE.get(variable, 1.0))
            offset = np.float32(self.OFFSET.get(variable, 0.0))
            values = raw.astype(np.float32) * scale + offset
            decoded[..., channel] = np.where(missing, np.float32(np.nan), values)
        return decoded

    def get_data_single(self, spatial_idx, time_idx):
        lat_index, lon_index = self.spatial_grid.rows_to_lat_lon_index(spatial_idx)
        stored = self.data_arr[int(time_idx), int(lat_index), int(lon_index)]
        return self.decode(np.asarray(stored))

    def get_data_multi(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
        lat_index, lon_index = self.spatial_grid.rows_to_lat_lon_index(indices[:, 0])
        stored = self.data_arr.vindex[indices[:, 1], lat_index, lon_index]
        return self.decode(stored)

    def shard_blocks(self, time_start, time_stop):
        """Split the time steps time_start .. time_stop - 1 at the shard boundaries.

        Returns:
            List of (block_start, block_stop); every block lies in one shard file.
        """
        steps_per_shard = self.data_arr.shards[0]
        blocks = []
        block_start = time_start
        while block_start < time_stop:
            next_shard_start = (block_start // steps_per_shard + 1) * steps_per_shard
            block_stop = min(next_shard_start, time_stop)
            blocks.append((block_start, block_stop))
            block_start = block_stop
        return blocks

    def get_data_series(self, spatial_indices, time_start, time_stop):
        """Reads one shard file after the other; within a shard, every inner chunk that holds
        one of the cells is decoded once, however many of the cells it holds."""
        spatial_indices = np.asarray(spatial_indices, dtype=np.int64)
        lat_index, lon_index = self.spatial_grid.rows_to_lat_lon_index(spatial_indices)
        n_steps = time_stop - time_start
        series = np.empty((spatial_indices.size, n_steps, len(self.variables)), dtype=np.float32)
        for block_start, block_stop in self.shard_blocks(time_start, time_stop):
            steps = np.arange(block_start, block_stop)
            stored = self.data_arr.vindex[steps[:, None], lat_index[None, :], lon_index[None, :]]
            decoded = self.decode(stored)  # (steps, cells, variables)
            series[:, block_start - time_start : block_stop - time_start] = decoded.swapaxes(0, 1)
        return series
