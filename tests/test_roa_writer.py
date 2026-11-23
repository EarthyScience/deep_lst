"""preprocessing.rain_over_africa.writer: the values of one shard from two synthetic raw years
with different source grids, across the turn of the year."""

import numpy as np
import pytest

from deep_lst.preprocessing.rain_over_africa.aggregation import RAIN_RATE_FILL, STORE_DTYPE
from deep_lst.preprocessing.rain_over_africa.mapping import TargetGrid, build_mapping
from deep_lst.preprocessing.rain_over_africa.writer import (
    RawYear,
    build_catalogue,
    shard_values,
    time_index_of_hours,
)

# rows at latitude 0.1, 0.05, 0, -0.05, -0.1 and columns at longitude 0, 0.05, ..., 0.4
SMALL_GRID = TargetGrid(lat_north=0.1, lon_west=0.0, resolution=0.05, n_lat=5, n_lon=9)
ALL_CELLS = np.arange(SMALL_GRID.n_cells)
SOURCE_SHAPE = (9, 16)
HOURS_PER_SHARD = 4


def source_coordinates():
    source_latitude = 0.1125 - 0.025 * np.arange(9)
    source_longitude = 0.0125 + 0.025 * np.arange(16)
    longitude, latitude = np.meshgrid(source_longitude, source_latitude)
    return latitude, longitude


def constant_scans(rates):
    """One scan per rate, the same rate in every source pixel. (n_scans, n_y, n_x)"""
    scans = np.empty((len(rates),) + SOURCE_SHAPE, dtype=np.float32)
    for scan_number, rate in enumerate(rates):
        scans[scan_number] = rate
    return scans


def year_2017():
    """Full source grid. One scan for the hour 23, four for the hour 00 of the next year."""
    latitude, longitude = source_coordinates()
    end_times = np.array(
        [
            "2017-12-31T22:57:41",
            "2017-12-31T23:12:41",
            "2017-12-31T23:27:41",
            "2017-12-31T23:42:41",
            "2017-12-31T23:57:41",
        ],
        dtype="datetime64[ms]",
    )
    scans = constant_scans([1.0, 2.0, np.inf, 4.0, 6.0])
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    return RawYear(year=2017, scan_end_times=end_times, rain_rate=scans, mapping=mapping)


def year_2018():
    """Source grid without the pixels of the first target row. Hour 01 with a duplicate scan,
    no scan for the hour 02, one scan for the hour 03."""
    latitude, longitude = source_coordinates()
    latitude[:2, :] = np.nan
    longitude[:2, :] = np.nan
    end_times = np.array(
        [
            "2018-01-01T00:12:41",
            "2018-01-01T00:27:41",
            "2018-01-01T00:42:41",
            "2018-01-01T00:42:44",  # second satellite, same slot
            "2018-01-01T00:57:41",
            "2018-01-01T02:12:41",
        ],
        dtype="datetime64[ms]",
    )
    scans = constant_scans([1.0, 1.0, 1.0, 99.0, 5.0, 7.0])
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    return RawYear(year=2018, scan_end_times=end_times, rain_rate=scans, mapping=mapping)


def test_time_index_counts_hours_since_1970():
    hours = np.array(["1970-01-01T05", "2016-01-01T00"], dtype="datetime64[h]")
    assert time_index_of_hours(hours).tolist() == [5, 16801 * 24]


def test_shard_holds_the_hours_of_both_years_with_their_own_mapping():
    raw_years = [year_2017(), year_2018()]
    catalogue = build_catalogue(raw_years)
    first_hour = np.datetime64("2017-12-31T23", "h")
    first_time_index = int(time_index_of_hours(first_hour))

    values, written = shard_values(raw_years, catalogue, first_time_index, HOURS_PER_SHARD)
    assert values.shape == (HOURS_PER_SHARD, 5, 9)
    assert values.dtype == STORE_DTYPE
    # hours 23, 00 and 01 have scans, hour 02 has none, hour 03 is outside the shard
    assert written.tolist() == [first_time_index, first_time_index + 1, first_time_index + 2]

    # hour 23: one scan of 1 mm/h
    assert (values["rain_rate"][0] == 100).all()
    assert (values["n_scans"][0] == 1).all()

    # hour 00 of 2018 comes from the 2017 store: 2, 4 and 6 mm/h, the scan with inf is left out
    assert (values["rain_rate"][1] == 400).all()
    assert (values["n_scans"][1] == 3).all()

    # hour 01 comes from the 2018 store: 1, 1, 1 and 5 mm/h, the duplicate scan is dropped;
    # the 2018 source grid has no pixels for the first target row
    assert (values["rain_rate"][2, 1:, :] == 200).all()
    assert (values["n_scans"][2, 1:, :] == 4).all()
    assert (values["rain_rate"][2, 0, :] == RAIN_RATE_FILL).all()
    assert (values["n_scans"][2, 0, :] == 0).all()

    # hour 02: no scan
    assert (values["rain_rate"][3] == RAIN_RATE_FILL).all()
    assert (values["n_scans"][3] == 0).all()


def test_hour_with_scans_of_two_years_is_rejected():
    first_year = year_2017()
    second_year = year_2018()
    end_times = second_year.scan_end_times.copy()
    end_times[0] = np.datetime64("2018-01-01T00:00:00", "ms")  # ends exactly on the hour 00
    second_year = RawYear(
        year=2018,
        scan_end_times=end_times,
        rain_rate=second_year.rain_rate,
        mapping=second_year.mapping,
    )
    raw_years = [first_year, second_year]
    catalogue = build_catalogue(raw_years)
    first_time_index = int(time_index_of_hours(np.datetime64("2017-12-31T23", "h")))
    with pytest.raises(ValueError):
        shard_values(raw_years, catalogue, first_time_index, HOURS_PER_SHARD)


def test_unsorted_raw_years_are_rejected():
    with pytest.raises(ValueError):
        build_catalogue([year_2018(), year_2017()])
