"""Statistics files and the normaliser, without eodata.

The fixture tests use tests/fixtures/readers_v1 (scripts/assets/export_reader_fixtures.py):
raw__<data type> is what the eodata readers returned, normalized__<data type> what the eodata
normaliser made of it, and statistics.json the statistics it used.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from deep_lst.utils.io import check_stats_transform, load_computed_stats, load_inc_variances
from deep_lst.utils.normalisation import StatsZarrNormalizer, VariableNormalizer, signed_expm1, signed_log1p
from deep_lst.utils.statistics import (
    increment_statistics,
    read_variable_statistics,
    variable_statistics,
    write_statistics,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "readers_v1"
META = json.loads((FIXTURE_DIR / "meta.json").read_text())
STATISTICS = json.loads((FIXTURE_DIR / "statistics.json").read_text())

# (modality, data type) of every normalised array in the fixtures
NORMALISED_CASES = [
    (modality, data_type)
    for modality, description in META["modalities"].items()
    for data_type in description["variables"]
    if data_type in STATISTICS[modality]["groups"]
]


def write_fixture_statistics(folder, modality):
    stats_zarr = folder / f"{modality}_stats.zarr"
    write_statistics(stats_zarr, STATISTICS[modality]["groups"], STATISTICS[modality]["transform"])
    return stats_zarr


@pytest.mark.parametrize(("modality", "data_type"), NORMALISED_CASES)
def test_normaliser_reproduces_the_eodata_pipeline(tmp_path, modality, data_type):
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    variables = META["modalities"][modality]["variables"][data_type]
    transform = STATISTICS[modality]["transform"] if data_type == "data" else None
    normalizer = VariableNormalizer(write_fixture_statistics(tmp_path, modality), variables, transform)

    normalised = normalizer.normalise(fixture[f"raw__{data_type}"])
    expected = fixture[f"normalized__{data_type}"]
    assert normalised.dtype == np.float32
    assert normalised.shape == expected.shape
    if transform is None:
        assert np.array_equal(normalised, expected, equal_nan=True)  # bit for bit
    else:  # numpy and torch round log1p differently in the last digit
        assert np.allclose(normalised, expected, rtol=0.0, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize(("modality", "data_type"), NORMALISED_CASES)
def test_denormalise_inverts_normalise(tmp_path, modality, data_type):
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    variables = META["modalities"][modality]["variables"][data_type]
    transform = STATISTICS[modality]["transform"] if data_type == "data" else None
    normalizer = VariableNormalizer(write_fixture_statistics(tmp_path, modality), variables, transform)

    raw = fixture[f"raw__{data_type}"]
    restored = normalizer.denormalise(normalizer.normalise(raw))
    largest = np.nanmax(np.abs(raw), axis=tuple(range(raw.ndim - 1)))  # per variable
    assert np.allclose(restored, raw, rtol=0.0, atol=1e-5 * np.maximum(largest, 1.0), equal_nan=True)


def test_normaliser_takes_the_variables_in_the_order_given(tmp_path):
    stats_zarr = write_fixture_statistics(tmp_path, "era5")
    normalizer = VariableNormalizer(stats_zarr, ["tcc", "sp"])
    names = STATISTICS["era5"]["groups"]["data"]["variables"]
    means = STATISTICS["era5"]["groups"]["data"]["mean"]
    assert normalizer.mean.tolist() == [np.float32(means[names.index("tcc")]), np.float32(means[names.index("sp")])]
    values = np.array([[means[names.index("tcc")], means[names.index("sp")]]])
    assert np.allclose(normalizer.normalise(values), 0.0, atol=1e-6)


def test_variables_of_all_value_groups_are_found(tmp_path):
    statistics = read_variable_statistics(write_fixture_statistics(tmp_path, "seviri"))
    assert {"LST", "FVC", "DEM"} <= set(statistics)
    mean, std = statistics["DEM"]
    assert mean == STATISTICS["seviri"]["groups"]["data_static"]["mean"][0]
    assert std == STATISTICS["seviri"]["groups"]["data_static"]["std"][0]


def test_a_zero_standard_deviation_does_not_divide_by_zero(tmp_path):
    groups = {"data": {"variables": ["flat"], "mean": [5.0], "std": [0.0], "min": [5.0], "max": [5.0]}}
    stats_zarr = tmp_path / "flat_stats.zarr"
    write_statistics(stats_zarr, groups)
    normalised = VariableNormalizer(stats_zarr, ["flat"]).normalise(np.array([[5.0], [np.nan]]))
    assert normalised[0, 0] == 0.0
    assert np.isnan(normalised[1, 0])


def test_unknown_variable_and_unknown_transform_are_rejected(tmp_path):
    stats_zarr = write_fixture_statistics(tmp_path, "era5")
    with pytest.raises(KeyError):
        VariableNormalizer(stats_zarr, ["not_a_variable"])
    with pytest.raises(ValueError, match="Unknown transform"):
        VariableNormalizer(stats_zarr, ["sp"], transform="square")


def test_signed_log1p_and_its_inverse():
    values = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    transformed = signed_log1p(values)
    assert transformed[2] == 0.0
    assert np.all(np.sign(transformed) == np.sign(values))
    assert np.allclose(transformed[3], np.log(2.0))
    assert np.allclose(signed_expm1(transformed), values, rtol=1e-5)


def test_variable_statistics_ignore_nan():
    values = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [6.0, np.nan]])
    statistics = variable_statistics(values)
    assert statistics["mean"].tolist() == [3.0, 20.0]
    assert np.allclose(statistics["std"], [np.std([1, 2, 3, 6], ddof=1), np.std([10, 30], ddof=1)])
    assert statistics["min"].tolist() == [1.0, 10.0]
    assert statistics["max"].tolist() == [6.0, 30.0]
    assert statistics["mean"].dtype == np.float64


def test_variable_statistics_are_taken_over_all_but_the_last_axis():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(7, 5, 24, 3)).astype(np.float32)
    statistics = variable_statistics(values)
    flat = values.reshape(-1, 3).astype(np.float64)
    assert np.allclose(statistics["mean"], flat.mean(axis=0))
    assert np.allclose(statistics["std"], flat.std(axis=0, ddof=1))


def test_increment_statistics_stay_within_the_time_axis():
    # two days of three hours for one cell and one variable: changes within a day are +1 and +2,
    # the jump from day to day (from 3 to 100) must not be counted
    values = np.array([[[[0.0], [1.0], [3.0]], [[100.0], [101.0], [103.0]]]])  # (1 cell, 2 days, 3 hours, 1)
    statistics = increment_statistics(values)
    assert statistics["mean"].tolist() == [1.5]
    assert statistics["min"].tolist() == [1.0]
    assert statistics["max"].tolist() == [2.0]


def test_written_file_is_read_by_the_existing_helpers(tmp_path):
    groups = {
        "data": {
            "variables": ["LST", "MLE"],
            "mean": [300.0, 50.0],
            "std": [10.0, 100.0],
            "min": [250, -5],
            "max": [340, 600],
        },
        "computed": {"variables": ["solar_zenith"], "mean": [90.0], "std": [45.0], "min": [0.0], "max": [180.0]},
        "increments": {
            "variables": ["LST"],
            "channels": [0],
            "mean": [0.1],
            "std": [3.0],
            "min": [-20.0],
            "max": [25.0],
        },
    }
    stats_zarr = tmp_path / "written_stats.zarr"
    write_statistics(stats_zarr, groups, transform="signed_log1p")

    check_stats_transform(stats_zarr, "signed_log1p")
    with pytest.raises(ValueError, match="transform"):
        check_stats_transform(stats_zarr, None)
    assert load_computed_stats(stats_zarr) == {"solar_zenith": (90.0, 45.0)}
    assert load_inc_variances(stats_zarr) == {"LST": pytest.approx((3.0 / 10.0) ** 2)}
    assert StatsZarrNormalizer(stats_zarr).denormalise(1.0, "LST") == 310.0
    assert read_variable_statistics(stats_zarr) == {"LST": (300.0, 10.0), "MLE": (50.0, 100.0)}


def test_file_without_a_transform_has_no_transform_attribute(tmp_path):
    stats_zarr = write_fixture_statistics(tmp_path, "era5")
    check_stats_transform(stats_zarr, None)
