"""CubeReader on the synthetic stores of tests/conftest.py.

A value of the synthetic stores is (field number + 1) * 1000 + (written position % 50) * 10
+ row % 10, so every expected value below can be worked out by hand.
"""

import numpy as np
import pytest

from deep_lst.stores import T1D, T1H, CubeReader, StoreReader

pytest.importorskip("zarr", minversion="3.4.2")

FIRST_HOUR = 403224  # 2016-01-01T00, the first written hour of hourly_store
FIRST_DAY = 16801  # 2016-01-01, the first written day of daily_store
N_LON = 8  # longitudes of the synthetic grid; the last one holds the fill value


class HourlyReader(CubeReader):
    """Like SeviriLSTReader: scaled integers, one variable with an offset."""

    MODALITY_ID = "synthetic_hourly"
    TIME_GRID = T1H
    SCALE = {"LST": 0.01, "standard_error": 0.01}
    OFFSET = {"LST": 273.15}


class DailyReader(CubeReader):
    MODALITY_ID = "synthetic_daily"
    TIME_GRID = T1D


class StaticReader(CubeReader):
    MODALITY_ID = "synthetic_static"


def test_reader_attributes(hourly_store):
    reader = HourlyReader(hourly_store.path)
    assert isinstance(reader, StoreReader)
    assert reader.modality_id == "synthetic_hourly"
    assert reader.variables == ["LST", "quality_flag", "standard_error"]
    assert reader.time_grid is T1H
    assert not reader.is_static
    assert reader.spatial_grid.shape == (6, 8)
    assert reader.data_arr.shape == (len(T1H), 6, 8)


def test_available_indices_are_the_written_steps(hourly_store):
    reader = HourlyReader(hourly_store.path)
    indices = reader.available_indices
    assert indices.shape == (len(hourly_store.written_steps), 2)
    assert np.all(indices[:, 0] == 0)
    assert np.array_equal(indices[:, 1], hourly_store.written_steps)


def test_get_data_single_scales_and_offsets(hourly_store):
    reader = HourlyReader(hourly_store.path)
    row = 2 * N_LON + 3  # lat index 2, lon index 3
    values = reader.get_data_single(row, FIRST_HOUR + 5)
    assert values.shape == (3,)
    assert values.dtype == np.float32
    # stored: LST 1000 + 50 + 9, quality_flag 2000 + 50 + 9, standard_error 3000 + 50 + 9
    expected = [1059 * 0.01 + 273.15, 2059.0, 3059 * 0.01]
    assert np.allclose(values, expected, rtol=1e-6)


def test_fill_value_becomes_nan(hourly_store):
    reader = HourlyReader(hourly_store.path)
    last_longitude = 2 * N_LON + (N_LON - 1)
    values = reader.get_data_single(last_longitude, FIRST_HOUR + 5)
    assert np.all(np.isnan(values))


def test_unwritten_steps_are_nan(hourly_store):
    reader = HourlyReader(hourly_store.path)
    row = 2 * N_LON + 3
    in_the_gap = FIRST_HOUR + 31
    before_the_record = FIRST_HOUR - 1
    in_a_missing_shard = FIRST_HOUR + 24 * 365
    for hour in (in_the_gap, before_the_record, in_a_missing_shard):
        assert np.all(np.isnan(reader.get_data_single(row, hour)))


def test_get_data_multi_keeps_the_order_of_the_pairs(hourly_store):
    reader = HourlyReader(hourly_store.path)
    pairs = [(19, FIRST_HOUR + 80), (3, FIRST_HOUR), (19, FIRST_HOUR + 1), (40, FIRST_HOUR + 100)]
    values = reader.get_data_multi(pairs)
    assert values.shape == (4, 3)
    assert values.dtype == np.float32
    for position, (row, hour) in enumerate(pairs):
        single = reader.get_data_single(row, hour)
        assert np.array_equal(values[position], single)


def test_get_data_multi_reads_one_cell_across_the_shard_boundary(hourly_store):
    reader = HourlyReader(hourly_store.path)
    row = 2 * N_LON + 3
    boundary = 4201 * 96
    hours = np.arange(boundary - 3, boundary + 3)
    pairs = np.stack([np.full(hours.size, row), hours], axis=1)
    quality_flag = reader.get_data_multi(pairs)[:, 1]
    positions = np.searchsorted(hourly_store.written_steps, hours)
    expected = hourly_store.values["quality_flag"][positions, 2, 3]
    assert np.array_equal(quality_flag, expected)


def test_get_data_multi_mixes_written_and_unwritten_steps(hourly_store):
    reader = HourlyReader(hourly_store.path)
    row = 2 * N_LON + 3
    hours = np.arange(FIRST_HOUR + 28, FIRST_HOUR + 35)  # the gap is hours 30, 31, 32
    pairs = [(row, int(hour)) for hour in hours]
    quality_flag = reader.get_data_multi(pairs)[:, 1]
    assert np.isnan(quality_flag).tolist() == [False, False, True, True, True, False, False]


def test_shard_blocks_split_a_time_range_at_the_shard_boundaries(hourly_store):
    reader = HourlyReader(hourly_store.path)
    boundary = 4201 * 96  # FIRST_HOUR is 24 hours into the shard 4200
    blocks = reader.shard_blocks(FIRST_HOUR, FIRST_HOUR + 5 * 24)
    assert blocks == [(FIRST_HOUR, boundary), (boundary, FIRST_HOUR + 5 * 24)]
    assert reader.shard_blocks(boundary, boundary + 96) == [(boundary, boundary + 96)]
    assert reader.shard_blocks(boundary + 5, boundary + 5) == []


def test_get_data_series_equals_get_data_multi(hourly_store):
    reader = HourlyReader(hourly_store.path)
    # cells of all four inner chunks, one of them twice, one on the fill-value longitude
    rows = np.array([0, 5, 3 * N_LON + 2, 5 * N_LON + 6, 5, 2 * N_LON + (N_LON - 1)])
    time_start, time_stop = FIRST_HOUR - 2, FIRST_HOUR + 5 * 24 + 2  # both shards, the gap, both ends
    series = reader.get_data_series(rows, time_start, time_stop)
    assert series.shape == (6, 5 * 24 + 4, 3)
    assert series.dtype == np.float32

    steps = np.arange(time_start, time_stop)
    for position, row in enumerate(rows):
        pairs = [(int(row), int(step)) for step in steps]
        expected = reader.get_data_multi(pairs)
        assert np.array_equal(series[position], expected, equal_nan=True)

    assert np.all(np.isnan(series[:, :2]))  # before the record
    assert np.all(np.isnan(series[5]))  # fill-value longitude
    assert not np.isnan(series[:5, 2:30]).any()  # written hours of the other cells


def test_get_data_series_of_a_missing_shard_is_nan(hourly_store):
    reader = HourlyReader(hourly_store.path)
    next_year = FIRST_HOUR + 24 * 365
    series = reader.get_data_series([3, 19], next_year, next_year + 200)
    assert series.shape == (2, 200, 3)
    assert np.all(np.isnan(series))


def test_get_data_series_of_a_daily_store_crosses_three_shards(daily_store):
    reader = DailyReader(daily_store.path)
    rows = [1 * N_LON + 2, 4 * N_LON + 5]
    series = reader.get_data_series(rows, FIRST_DAY, FIRST_DAY + 10)
    assert series.shape == (2, 10, 2)
    # dis24 of the cell with row 10: 1000 + 10 * day + 0
    assert series[0, :, 0].tolist() == [1000.0 + 10 * day for day in range(10)]
    expected = daily_store.values["swir"][:, 4, 5]
    assert np.array_equal(series[1, :, 1], expected)


def test_get_data_series_of_a_static_store(static_store):
    reader = StaticReader(static_store.path)
    series = reader.get_data_series([0, 9], 0, 1)
    assert series.shape == (2, 1, 1)
    assert series[:, 0, 0].tolist() == [1000.0, 1009.0]


def test_spatial_chunks_cover_the_grid_once(hourly_store):
    reader = HourlyReader(hourly_store.path)
    rectangles = reader.spatial_chunks()
    assert len(rectangles) == 4  # the synthetic grid has 2 x 2 inner chunks of 3 x 4 cells
    assert rectangles[0] == (slice(0, 3), slice(0, 4))
    assert rectangles[3] == (slice(3, 6), slice(4, 8))
    covered = np.zeros(reader.spatial_grid.shape, dtype=int)
    for lat_slice, lon_slice in rectangles:
        covered[lat_slice, lon_slice] += 1
    assert np.all(covered == 1)


def test_get_data_box_equals_get_data_series(hourly_store):
    reader = HourlyReader(hourly_store.path)
    lat_slice, lon_slice = slice(1, 5), slice(2, 8)  # across all four inner chunks, with the fill longitude
    time_start, time_stop = FIRST_HOUR + 90, FIRST_HOUR + 100  # across the shard boundary
    box = reader.get_data_box(lat_slice, lon_slice, time_start, time_stop)
    assert box.shape == (10, 4, 6, 3)
    assert box.dtype == np.float32

    lat_index, lon_index = np.meshgrid(np.arange(1, 5), np.arange(2, 8), indexing="ij")
    rows = reader.spatial_grid.lat_lon_index_to_rows(lat_index.ravel(), lon_index.ravel())
    series = reader.get_data_series(rows, time_start, time_stop)  # (cells, steps, variables)
    expected = series.reshape(4, 6, 10, 3).transpose(2, 0, 1, 3)
    assert np.array_equal(box, expected, equal_nan=True)
    assert np.all(np.isnan(box[:, :, -1]))  # the fill longitude


def test_reader_refuses_a_store_of_another_modality(daily_store):
    with pytest.raises(ValueError, match="synthetic_daily"):
        HourlyReader(daily_store.path)


def test_daily_store_with_float_fields(daily_store):
    reader = DailyReader(daily_store.path)
    assert reader.time_grid is T1D
    assert reader.variables == ["dis24", "swir"]
    row = 1 * N_LON + 2
    values = reader.get_data_single(row, FIRST_DAY + 4)
    # stored: dis24 1000 + 40 + 0, swir 2000 + 40 + 0; no scale
    assert values.tolist() == [1040.0, 2040.0]
    assert np.all(np.isnan(reader.get_data_single(row, FIRST_DAY - 1)))
    assert np.all(np.isnan(reader.get_data_single(N_LON - 1, FIRST_DAY + 4)))


def test_static_store_has_the_time_index_zero(static_store):
    reader = StaticReader(static_store.path)
    assert reader.is_static
    assert reader.time_grid is None
    assert reader.available_indices.tolist() == [[0, 0]]
    rows = [0, 9, N_LON - 1]
    values = reader.get_data_multi([(row, 0) for row in rows])
    assert values.shape == (3, 1)
    assert values[:2, 0].tolist() == [1000.0, 1009.0]
    assert np.isnan(values[2, 0])
