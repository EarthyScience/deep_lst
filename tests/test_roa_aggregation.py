"""preprocessing.rain_over_africa.aggregation: cleaning of the raw rates, the hour a scan
belongs to, duplicate scans, the hourly mean and its integer encoding."""

import numpy as np

from deep_lst.preprocessing.rain_over_africa.aggregation import (
    RAIN_RATE_FILL,
    STORE_DTYPE,
    STORE_FILL_VALUE,
    clean_rates,
    encode_rain_rate,
    first_scan_per_slot,
    group_scans_by_hour,
    hour_labels,
    hourly_mean_rate,
)


def times(*texts):
    return np.array(texts, dtype="datetime64[ms]")


def test_clean_rates_keeps_plausible_rain_and_drops_artefacts():
    raw = np.array([0.0, 3.5, 200.0, 200.1, 5e20, np.inf, np.nan, -1e-11, -0.5], dtype=np.float32)
    clean, valid = clean_rates(raw)
    assert valid.tolist() == [True, True, True, False, False, False, False, True, False]
    assert np.allclose(clean, [0.0, 3.5, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert clean.dtype == np.float32


def test_scan_belongs_to_the_hour_it_ends_in():
    end_times = times(
        "2020-08-15T13:12:43.349",
        "2020-08-15T13:57:43.350",
        "2020-08-15T14:00:00.000",
        "2020-08-15T14:00:00.001",
        "2020-08-15T14:12:41.000",
        "2020-12-31T23:57:43.960",
    )
    expected = np.array(
        ["2020-08-15T14", "2020-08-15T14", "2020-08-15T14", "2020-08-15T15", "2020-08-15T15", "2021-01-01T00"],
        dtype="datetime64[h]",
    )
    assert np.array_equal(hour_labels(end_times), expected)


def test_second_scan_of_a_slot_is_dropped():
    end_times = times(
        "2016-03-01T10:12:40.824",
        "2016-03-01T10:12:43.028",  # second satellite, same slot
        "2016-03-01T10:27:41.000",
        "2016-03-01T10:42:42.000",
    )
    assert first_scan_per_slot(end_times).tolist() == [True, False, True, True]


def test_scans_are_grouped_by_hour_without_duplicates():
    end_times = times(
        "2016-03-01T09:57:41.000",  # hour 10
        "2016-03-01T10:12:40.824",  # hour 11
        "2016-03-01T10:12:43.028",  # hour 11, duplicate slot
        "2016-03-01T10:27:41.000",  # hour 11
        "2016-03-01T10:42:42.000",  # hour 11
        "2016-03-01T10:57:42.000",  # hour 11
        "2016-03-01T12:12:42.000",  # hour 13, hour 12 has no scan
    )
    hours, scan_positions = group_scans_by_hour(end_times)
    expected_hours = np.array(["2016-03-01T10", "2016-03-01T11", "2016-03-01T13"], dtype="datetime64[h]")
    assert np.array_equal(hours, expected_hours)
    assert [positions.tolist() for positions in scan_positions] == [[0], [1, 3, 4, 5], [6]]


def test_hourly_mean_uses_only_valid_scans():
    scan_rates = np.array(
        [
            [4.0, 1.0, 0.0, 9.0],
            [0.0, 1.0, 0.0, 9.0],
            [0.0, 3.0, 0.0, 9.0],
            [0.0, 3.0, 0.0, 9.0],
        ]
    )
    scan_valid = np.array(
        [
            [True, True, False, True],
            [True, True, False, False],
            [True, False, False, False],
            [True, False, False, False],
        ]
    )
    mean_rate, n_scans = hourly_mean_rate(scan_rates, scan_valid)
    # cell 0: 4 mm/h for a quarter of an hour = 1 mm; cell 1: mean of two scans; cell 2: no scan
    assert np.allclose(mean_rate, [1.0, 1.0, 0.0, 9.0])
    assert n_scans.tolist() == [4, 2, 0, 1]
    assert mean_rate.dtype == np.float32
    assert n_scans.dtype == np.int8


def test_encoding_rounds_to_hundredths_and_marks_missing_cells():
    mean_rate = np.array([0.0, 0.004, 0.005001, 1.237, 200.0, 0.0], dtype=np.float32)
    n_scans = np.array([4, 4, 4, 3, 1, 0], dtype=np.int8)
    encoded = encode_rain_rate(mean_rate, n_scans)
    assert encoded.tolist() == [0, 0, 1, 124, 20000, RAIN_RATE_FILL]
    assert encoded.dtype == np.int16


def test_store_dtype_matches_the_fill_values():
    assert STORE_DTYPE.names == ("rain_rate", "n_scans")
    assert STORE_DTYPE.itemsize == 3
    assert len(STORE_FILL_VALUE) == len(STORE_DTYPE.names)
