"""TimeAxis: the time index of the product stores, without eodata."""

import numpy as np
import pandas as pd
import pytest

from deep_lst.stores import T1D, T1H, TimeAxis


def test_axis_lengths_are_the_time_dimensions_of_the_stores():
    assert len(T1H) == 613608
    assert len(T1D) == 25567


def test_start_is_1970_in_utc():
    assert T1H.start == pd.Timestamp("1970-01-01T00:00", tz="UTC")
    assert T1D.start == T1H.start


def test_date_to_index_known_values():
    # the constants of deep_lst/dataset/dataset.py
    assert T1H.date_to_index("2016-01-01") == 403224
    assert T1H.date_to_index("2016-01-01") // 24 == 16801
    assert T1D.date_to_index("2016-01-01") == 16801
    assert T1H.date_to_index("1970-01-01") == 0
    assert T1H.date_to_index("2039-12-31T23:00") == len(T1H) - 1


def test_date_inside_a_step_gives_that_step():
    assert T1H.date_to_index("2016-01-01T05:59:59") == 403224 + 5
    assert T1D.date_to_index("2016-01-01T23:00") == 16801


def test_date_without_time_zone_is_utc():
    naive = pd.Timestamp("2024-02-29T12:00")
    utc = pd.Timestamp("2024-02-29T12:00", tz="UTC")
    berlin = pd.Timestamp("2024-02-29T13:00", tz="Europe/Berlin")
    assert T1H.date_to_index(naive) == T1H.date_to_index(utc)
    assert T1H.date_to_index(berlin) == T1H.date_to_index(utc)


@pytest.mark.parametrize("date", ["1969-12-31T23:00", "2040-01-01T00:00"])
def test_date_outside_the_axis_raises(date):
    with pytest.raises(KeyError):
        T1H.date_to_index(date)


def test_index_to_startdate_of_one_index():
    start = T1H.index_to_startdate(403224)
    assert start == pd.Timestamp("2016-01-01T00:00", tz="UTC")
    assert T1D.index_to_startdate(16801) == start


def test_index_to_startdate_of_an_array():
    hours = np.array([403224, 403225, 403224 + 24 * 366])
    starts = T1H.index_to_startdate(hours)
    assert isinstance(starts, pd.DatetimeIndex)
    assert str(starts.tz) == "UTC"
    expected = pd.DatetimeIndex(["2016-01-01T00:00", "2016-01-01T01:00", "2017-01-01T00:00"], tz="UTC")
    assert starts.equals(expected)


def test_round_trip_over_a_leap_year():
    hours = np.arange(T1H.date_to_index("2024-01-01"), T1H.date_to_index("2025-01-01"))
    assert hours.size == 366 * 24
    starts = T1H.index_to_startdate(hours)
    back = [T1H.date_to_index(start) for start in starts[::997]]
    assert back == hours[::997].tolist()


def test_startdates_match_the_time_idx_array_of_a_store():
    # a store keeps its axis as datetime64[ns] from 1970-01-01T00 in steps of one hour
    hours = np.array([0, 1, 403224, len(T1H) - 1])
    stored = np.datetime64("1970-01-01T00", "ns") + hours * np.timedelta64(1, "h")
    starts = T1H.index_to_startdate(hours).tz_localize(None)
    assert np.array_equal(starts.to_numpy(), stored)


def test_a_new_axis_takes_any_step():
    three_hourly = TimeAxis("T3H", "3h")
    assert len(three_hourly) == 613608 // 3
    assert three_hourly.date_to_index("1970-01-01T05:00") == 1
