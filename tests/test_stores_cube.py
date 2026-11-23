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
