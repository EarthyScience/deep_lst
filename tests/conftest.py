"""Synthetic stores in the layout of the product stores, for the tests of deep_lst.stores.

A product store is a zarr v3 group with five arrays:

    data      (time, lat, lon)  struct dtype, one field per variable; one shard holds
                                steps_per_shard time steps of the whole grid
    lat       (n_lat,)          float64, north to south
    lon       (n_lon,)          float64, west to east
    time_idx  (time,)           datetime64[ns], start of every step of the time axis
    _indices  (n_written, 2)    int64, one row (0, time index) per written time step

The time dimension spans the whole axis (1970 to 2040); only the shards that were written exist
as files. A static store has one time step. Array metadata and codecs are those of the stores
in data/deep_lst as read on 2026-11-23. The root attributes follow EOZarr_v2.1; v2.2 stores
nest spatial_grid and time_grid in a "zarr_schema" attribute instead.

Sharded struct arrays need zarr >= 3.4.2: tests that use these fixtures are skipped below it.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from deep_lst.stores import T1D, T1H, LatLonGrid, TimeAxis

FIRST_HOUR_2016 = 403224  # 2016-01-01T00, 24 hours into the shard 4200 of an hourly store
FIRST_DAY_2016 = 16801  # 2016-01-01, one day into the shard 4200 of a daily store

# modality -> first latitude, first longitude, resolution, n_lat, n_lon of the grid of its stores
STORE_GRID_SPECS = {
    "seviri": (80.0, -80.0, 0.05, 3201, 3201),
    "era5": (90.0, -180.0, 0.25, 721, 1440),
    "era5_land": (90.0, -180.0, 0.1, 1801, 3600),
    "glofas_v4": (89.975, -179.975, 0.05, 3000, 7200),
}


@pytest.fixture(scope="session")
def store_grids():
    """{modality: LatLonGrid} of the real stores, built from their extent instead of read from them."""
    grids = {}
    for modality, (lat_first, lon_first, resolution, n_lat, n_lon) in STORE_GRID_SPECS.items():
        lat = lat_first - resolution * np.arange(n_lat)
        lon = lon_first + resolution * np.arange(n_lon)
        grids[modality] = LatLonGrid(lat, lon, name=modality)
    return grids


@dataclass
class SyntheticStore:
    """A store written by write_product_store and what went into it."""

    path: Path
    time_axis: TimeAxis | None  # None: static store
    lat: np.ndarray
    lon: np.ndarray
    fill_value: tuple
    steps_per_shard: int
    written_steps: np.ndarray  # (n_written,) time indices
    values: np.ndarray  # (n_written, n_lat, n_lon) struct array, as written


def synthetic_values(written_steps, n_lat, n_lon, dtype):
    """Values that encode where they are: field number, time step and cell.

    Field k of the cell with row r at the i-th written step holds
    (k + 1) * 1000 + (i % 50) * 10 + r % 10.
    """
    n_written = len(written_steps)
    step_part = (np.arange(n_written) % 50) * 10
    cell_part = np.arange(n_lat * n_lon).reshape(n_lat, n_lon) % 10
    values = np.zeros((n_written, n_lat, n_lon), dtype=dtype)
    for field_number, field in enumerate(dtype.names):
        field_part = (field_number + 1) * 1000
        values[field] = field_part + step_part[:, None, None] + cell_part[None, :, :]
    return values


def write_product_store(path, modality_id, time_axis, lat, lon, steps_per_shard, written_steps, values, fill_value):
    """Write a store in the layout of the product stores.

    Args:
        path: Folder of the store (<name>.zarr).
        modality_id: Written to the root attributes.
        time_axis: TimeAxis of the store; None writes a static store with one time step.
        lat: Latitudes, north to south. Length even, the grid is split into 2 x 2 inner chunks.
        lon: Longitudes, west to east. Length even.
        steps_per_shard: Time steps in one shard file.
        written_steps: Time indices that receive values.
        values: Struct array (len(written_steps), n_lat, n_lon).
        fill_value: One fill value per field.

    Returns:
        The SyntheticStore describing what was written.
    """
    zarr = pytest.importorskip("zarr", minversion="3.4.2")
    from zarr.codecs import BloscCodec, Crc32cCodec

    n_lat, n_lon = len(lat), len(lon)
    written_steps = np.asarray(written_steps, dtype=np.int64)
    if time_axis is None:
        time_values = np.array(["1970-01-01T00"], dtype="datetime64[ns]")
        time_grid_name = "TSTATIC"
    else:
        time_values = time_axis.index_to_startdate(np.arange(len(time_axis))).tz_localize(None).to_numpy()
        time_grid_name = time_axis.name
    n_steps = len(time_values)

    separator = {"name": "default", "separator": "."}
    attributes = {
        "EOZarr_version": "EOZarr_v2.1",
        "modality_id": modality_id,
        "spatial_grid": "synthetic",
        "time_grid": time_grid_name,
    }
    root = zarr.open_group(str(path), mode="w", zarr_format=3, attributes=attributes)

    data = root.create_array(
        "data",
        shape=(n_steps, n_lat, n_lon),
        dtype=values.dtype,
        shards=(steps_per_shard, n_lat, n_lon),
        chunks=(steps_per_shard, n_lat // 2, n_lon // 2),
        fill_value=np.array(fill_value, dtype=values.dtype)[()],
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"), Crc32cCodec()],
        dimension_names=["time_idx", "lat", "lon"],
        chunk_key_encoding=separator,
    )
    for position, step in enumerate(written_steps):
        data[int(step)] = values[position]

    coordinates = {"lat": np.asarray(lat, dtype=np.float64), "lon": np.asarray(lon, dtype=np.float64)}
    for name, coordinate in coordinates.items():
        array = root.create_array(
            name,
            shape=coordinate.shape,
            dtype="float64",
            chunks=coordinate.shape,
            fill_value=np.nan,
            dimension_names=[name],
            chunk_key_encoding=separator,
        )
        array[:] = coordinate

    time_idx = root.create_array(
        "time_idx",
        shape=(n_steps,),
        dtype="datetime64[ns]",
        chunks=(n_steps,),
        dimension_names=["time_idx"],
        chunk_key_encoding=separator,
    )
    time_idx[:] = time_values

    index_rows = np.zeros((len(written_steps), 2), dtype=np.int64)
    index_rows[:, 1] = written_steps
    indices = root.create_array(
        "_indices",
        shape=index_rows.shape,
        dtype="int64",
        chunks=(100000, 2),
        fill_value=-1,
        compressors=None,
        dimension_names=["sample", "axis"],
        chunk_key_encoding=separator,
    )
    indices[:] = index_rows

    return SyntheticStore(
        path=Path(path),
        time_axis=time_axis,
        lat=coordinates["lat"],
        lon=coordinates["lon"],
        fill_value=tuple(fill_value),
        steps_per_shard=steps_per_shard,
        written_steps=written_steps,
        values=values,
    )


def synthetic_grid():
    """Six latitudes and eight longitudes at half a degree."""
    lat = 10.0 - 0.5 * np.arange(6)
    lon = 20.0 + 0.5 * np.arange(8)
    return lat, lon


@pytest.fixture
def hourly_store(tmp_path):
    """Hourly store like seviri_product_hourly_lst: int16 fields, shards of 96 hours.

    Five days from 2016-01-01T00 are written, over the shards 4200 and 4201, except three hours
    on the second day (the gap). The last longitude holds the fill value at every step, like a
    cell that is not on the land grid.
    """
    lat, lon = synthetic_grid()
    dtype = np.dtype([("LST", "<i2"), ("quality_flag", "<i2"), ("standard_error", "<i2")])
    fill_value = (-32768, -9999, -1)
    hours = np.arange(FIRST_HOUR_2016, FIRST_HOUR_2016 + 5 * 24)
    gap = np.arange(FIRST_HOUR_2016 + 30, FIRST_HOUR_2016 + 33)
    written_steps = np.setdiff1d(hours, gap)
    values = synthetic_values(written_steps, len(lat), len(lon), dtype)
    values[:, :, -1] = np.array(fill_value, dtype=dtype)
    return write_product_store(
        tmp_path / "synthetic_hourly.zarr", "synthetic_hourly", T1H, lat, lon, 96, written_steps, values, fill_value
    )


@pytest.fixture
def daily_store(tmp_path):
    """Daily store like glofas_v4_product: float32 fields with NaN as fill value, shards of 4 days.

    Ten days from 2016-01-01 are written, over the shards 4200 to 4202. The last longitude is NaN.
    """
    lat, lon = synthetic_grid()
    dtype = np.dtype([("dis24", "<f4"), ("swir", "<f4")])
    fill_value = (np.nan, np.nan)
    written_steps = np.arange(FIRST_DAY_2016, FIRST_DAY_2016 + 10)
    values = synthetic_values(written_steps, len(lat), len(lon), dtype)
    values[:, :, -1] = np.array(fill_value, dtype=dtype)
    return write_product_store(
        tmp_path / "synthetic_daily.zarr", "synthetic_daily", T1D, lat, lon, 4, written_steps, values, fill_value
    )


@pytest.fixture
def static_store(tmp_path):
    """Static store like seviri_product_static_dem: one time step, one int16 field."""
    lat, lon = synthetic_grid()
    dtype = np.dtype([("DEM", "<i2")])
    fill_value = (-9999,)
    written_steps = np.array([0])
    values = synthetic_values(written_steps, len(lat), len(lon), dtype)
    values[:, :, -1] = np.array(fill_value, dtype=dtype)
    return write_product_store(
        tmp_path / "synthetic_static.zarr", "synthetic_static", None, lat, lon, 1, written_steps, values, fill_value
    )


@pytest.fixture
def class_store(tmp_path):
    """Static store like seviri_product_static_landcov: one int8 class field with the fill value 100.

    The class of the cell with row r is r % 5; the last longitude holds the fill value.
    """
    lat, lon = synthetic_grid()
    dtype = np.dtype([("LANDCOV", "i1")])
    fill_value = (100,)
    written_steps = np.array([0])
    values = np.zeros((1, len(lat), len(lon)), dtype=dtype)
    values["LANDCOV"][0] = np.arange(len(lat) * len(lon)).reshape(len(lat), len(lon)) % 5
    values[:, :, -1] = np.array(fill_value, dtype=dtype)
    return write_product_store(
        tmp_path / "synthetic_class.zarr", "synthetic_class", None, lat, lon, 1, written_steps, values, fill_value
    )


@pytest.fixture
def product_store_writer():
    """write_product_store, for tests that need a store the fixtures above do not provide."""
    return write_product_store
