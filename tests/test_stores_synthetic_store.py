"""The synthetic stores of tests/conftest.py have the layout of the product stores.

Read back with plain zarr, so these tests also pin what the readers will find on disk.
"""

import json

import numpy as np
import pytest

from deep_lst.stores import T1H, LatLonGrid

zarr = pytest.importorskip("zarr", minversion="3.4.2")


def array_metadata(store, name):
    return json.loads((store.path / name / "zarr.json").read_text())


def test_store_has_the_five_arrays(hourly_store):
    entries = sorted(entry.name for entry in hourly_store.path.iterdir())
    assert entries == ["_indices", "data", "lat", "lon", "time_idx", "zarr.json"]


def test_data_array_metadata_is_that_of_a_product_store(hourly_store):
    metadata = array_metadata(hourly_store, "data")
    assert metadata["shape"] == [len(T1H), 6, 8]
    assert metadata["data_type"]["name"] == "struct"
    field_names = [field["name"] for field in metadata["data_type"]["configuration"]["fields"]]
    assert field_names == ["LST", "quality_flag", "standard_error"]
    assert metadata["fill_value"] == {"LST": -32768, "quality_flag": -9999, "standard_error": -1}
    assert metadata["dimension_names"] == ["time_idx", "lat", "lon"]
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [96, 6, 8]
    assert metadata["chunk_key_encoding"]["configuration"]["separator"] == "."

    sharding = metadata["codecs"][0]
    assert sharding["name"] == "sharding_indexed"
    assert sharding["configuration"]["chunk_shape"] == [96, 3, 4]
    inner_codecs = [codec["name"] for codec in sharding["configuration"]["codecs"]]
    assert inner_codecs == ["bytes", "blosc", "crc32c"]


def test_only_the_written_shards_are_files(hourly_store):
    shard_files = sorted(entry.name for entry in (hourly_store.path / "data").iterdir())
    assert shard_files == ["c.4200.0.0", "c.4201.0.0", "zarr.json"]


def test_written_steps_read_back(hourly_store):
    data = zarr.open_array(str(hourly_store.path / "data"), mode="r")
    first, last = int(hourly_store.written_steps[0]), int(hourly_store.written_steps[-1])
    stored = data[first : last + 1]
    positions = hourly_store.written_steps - first
    assert np.array_equal(stored[positions], hourly_store.values)


def test_unwritten_steps_hold_the_fill_value(hourly_store):
    data = zarr.open_array(str(hourly_store.path / "data"), mode="r")
    first = int(hourly_store.written_steps[0])
    in_the_gap = first + 31
    assert in_the_gap not in hourly_store.written_steps
    for step in (in_the_gap, first - 1, 0):
        values = data[step]
        assert np.all(values["LST"] == -32768)
        assert np.all(values["quality_flag"] == -9999)
        assert np.all(values["standard_error"] == -1)


def test_one_cell_over_time_crosses_the_shard_boundary(hourly_store):
    data = zarr.open_array(str(hourly_store.path / "data"), mode="r")
    boundary = 4201 * 96
    series = data[boundary - 2 : boundary + 2, 2, 3]["LST"]
    positions = np.searchsorted(hourly_store.written_steps, np.arange(boundary - 2, boundary + 2))
    expected = hourly_store.values["LST"][positions, 2, 3]
    assert np.array_equal(series, expected)


def test_indices_list_the_written_steps(hourly_store):
    indices = zarr.open_array(str(hourly_store.path / "_indices"), mode="r")[:]
    assert indices.shape == (len(hourly_store.written_steps), 2)
    assert np.all(indices[:, 0] == 0)
    assert np.array_equal(indices[:, 1], hourly_store.written_steps)


def test_time_idx_is_the_time_axis(hourly_store):
    time_idx = zarr.open_array(str(hourly_store.path / "time_idx"), mode="r")
    assert time_idx.shape == (len(T1H),)
    for hour in (0, 403224, len(T1H) - 1):
        start = T1H.index_to_startdate(hour).tz_localize(None)
        assert time_idx[hour] == start.to_datetime64()


def test_lat_and_lon_give_a_grid(hourly_store):
    lat = zarr.open_array(str(hourly_store.path / "lat"), mode="r")[:]
    lon = zarr.open_array(str(hourly_store.path / "lon"), mode="r")[:]
    grid = LatLonGrid(lat, lon, name="synthetic")
    assert grid.shape == (6, 8)
    assert grid.lat_step == -0.5
    assert grid.lon_step == 0.5


def test_daily_store_has_float_fields_and_nan_fill(daily_store):
    metadata = array_metadata(daily_store, "data")
    assert metadata["shape"] == [25567, 6, 8]
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [4, 6, 8]
    data = zarr.open_array(str(daily_store.path / "data"), mode="r")
    first = int(daily_store.written_steps[0])
    stored = data[first : first + 10]
    assert np.array_equal(stored["dis24"], daily_store.values["dis24"], equal_nan=True)
    assert np.all(np.isnan(stored["dis24"][:, :, -1]))
    assert np.all(np.isnan(data[first - 1]["swir"]))
    shard_files = sorted(entry.name for entry in (daily_store.path / "data").iterdir())
    assert shard_files == ["c.4200.0.0", "c.4201.0.0", "c.4202.0.0", "zarr.json"]


def test_static_store_has_one_time_step(static_store):
    data = zarr.open_array(str(static_store.path / "data"), mode="r")
    assert data.shape == (1, 6, 8)
    assert np.array_equal(data[0], static_store.values[0])
    time_idx = zarr.open_array(str(static_store.path / "time_idx"), mode="r")[:]
    assert time_idx.tolist() == [0]
    indices = zarr.open_array(str(static_store.path / "_indices"), mode="r")[:]
    assert indices.tolist() == [[0, 0]]
