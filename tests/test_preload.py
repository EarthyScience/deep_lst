"""The preload of the product stores, on the synthetic stores of tests/conftest.py.

A value of the synthetic stores is (field number + 1) * 1000 + (written position % 50) * 10
+ row % 10; the hourly store has a gap of three hours (30 to 32 after its start), and the last
longitude of every store holds the fill value.
"""

import numpy as np
import pytest
import torch

from deep_lst import stores
from deep_lst.dataset import preload
from deep_lst.dataset.preload import preload_modalities, preload_modality, variable_stores
from deep_lst.stores import T1D, T1H, CubeReader, LatLonGrid
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.statistics import write_statistics

pytest.importorskip("zarr", minversion="3.4.2")

FIRST_DAY = 16801  # 2016-01-01, the first written day of the synthetic stores
N_LON = 8
FIVE_DAYS = (FIRST_DAY, FIRST_DAY + 5)

SOURCE = {
    "stores": {
        "synthetic_hourly": ["LST", "quality_flag"],
        "synthetic_daily": ["dis24"],
        "synthetic_static": ["DEM"],
        "synthetic_class": ["LANDCOV"],
    },
    "load": {
        "data": ["LST"],
        "data_daily": ["dis24"],
        "data_static": ["DEM"],
        "data_static_categorical": ["LANDCOV"],
        "metadata": ["quality_flag"],
    },
}


class HourlyReader(CubeReader):
    MODALITY_ID = "synthetic_hourly"
    TIME_GRID = T1H


class DailyReader(CubeReader):
    MODALITY_ID = "synthetic_daily"
    TIME_GRID = T1D


class StaticReader(CubeReader):
    MODALITY_ID = "synthetic_static"


class ClassReader(CubeReader):
    MODALITY_ID = "synthetic_class"


@pytest.fixture
def paths(tmp_path, monkeypatch, hourly_store, daily_store, static_store, class_store):
    """ExperimentPaths over the four synthetic stores, whose readers are registered."""
    for reader_class in (HourlyReader, DailyReader, StaticReader, ClassReader):
        monkeypatch.setitem(stores.READERS, reader_class.MODALITY_ID, reader_class)
    return ExperimentPaths(product_root=tmp_path, assets_root=tmp_path / "assets")


@pytest.fixture
def grid(hourly_store):
    return LatLonGrid(hourly_store.lat, hourly_store.lon, name="synthetic")


def written_position(hour):
    """Position of an hour of the first five days among the written hours of hourly_store."""
    return hour if hour < 30 else hour - 3


def test_leaves_have_the_layout_of_the_old_preload(paths, grid):
    anchor_ids = np.array([3, 19, 40])
    columns, inverse, cell_lonlat = preload_modality(
        SOURCE, "synthetic", anchor_ids, paths, anchor_grid=grid, day_range=FIVE_DAYS
    )

    assert set(columns) == set(SOURCE["load"])
    assert all(isinstance(values, torch.Tensor) for values in columns.values())
    assert columns["data"].shape == (3, 120, 1)
    assert columns["data_daily"].shape == (3, 120, 1)
    assert columns["metadata"].shape == (3, 120, 1)
    assert columns["data_static"].shape == (3, 1)
    assert columns["data_static_categorical"].shape == (3, 1)
    assert columns["data"].dtype == torch.float32
    assert columns["data_static_categorical"].dtype == torch.int8
    assert inverse.tolist() == [0, 1, 2]
    assert np.array_equal(cell_lonlat, grid.centers[anchor_ids])


def test_hourly_values_and_missing_hours(paths, grid):
    columns, _, _ = preload_modality(SOURCE, "synthetic", [19], paths, anchor_grid=grid, day_range=FIVE_DAYS)
    lst = columns["data"][0, :, 0].numpy()
    quality_flag = columns["metadata"][0, :, 0].numpy()

    assert np.isnan(lst[30:33]).all()  # the gap
    for hour in (0, 29, 33, 95, 96, 119):  # 95 and 96 are the two sides of the shard boundary
        position = written_position(hour)
        assert lst[hour] == 1000 + (position % 50) * 10 + 19 % 10
        assert quality_flag[hour] == 2000 + (position % 50) * 10 + 19 % 10
    assert np.isnan(lst).sum() == 3


def test_daily_values_are_repeated_over_their_day(paths, grid):
    columns, _, _ = preload_modality(SOURCE, "synthetic", [19], paths, anchor_grid=grid, day_range=FIVE_DAYS)
    daily = columns["data_daily"][0, :, 0].numpy().reshape(5, 24)
    for day in range(5):
        assert np.all(daily[day] == 1000 + day * 10 + 19 % 10)


def test_static_values_and_classes(paths, grid):
    fill_longitude = 2 * N_LON + (N_LON - 1)
    columns, _, _ = preload_modality(
        SOURCE, "synthetic", [3, 19, fill_longitude], paths, anchor_grid=grid, day_range=FIVE_DAYS
    )
    dem = columns["data_static"][:, 0].numpy()
    classes = columns["data_static_categorical"][:, 0].numpy()
    assert dem[:2].tolist() == [1003.0, 1009.0]
    assert np.isnan(dem[2])
    assert classes.tolist() == [3 % 5, 19 % 5, -1]  # a missing class is -1
    assert torch.isnan(columns["data"][2]).all()


def test_anchors_sharing_a_cell_are_read_once(paths, grid):
    # anchors on a grid twice as fine: four of them fall into one cell of the store
    fine_lat = 10.0 - 0.25 * np.arange(12)
    fine_lon = 20.0 + 0.25 * np.arange(16)
    fine = LatLonGrid(fine_lat, fine_lon, name="fine")
    anchor_ids = fine.lat_lon_index_to_rows([4, 4, 5, 0], [6, 7, 6, 0])  # three near (9.0, 21.5), one at (10, 20)
    columns, inverse, cell_lonlat = preload_modality(
        SOURCE, "fine", anchor_ids, paths, anchor_grid=fine, day_range=FIVE_DAYS
    )
    assert columns["data"].shape[0] == len(np.unique(inverse))
    assert inverse[0] == inverse[1] or inverse[0] == inverse[2]
    per_anchor = columns["data_static"][inverse][:, 0].numpy()
    rows = grid.nearest_rows(fine.lon[[6, 7, 6, 0]], fine.lat[[4, 4, 5, 0]])
    assert per_anchor.tolist() == [1000.0 + row % 10 for row in rows]
    assert np.array_equal(cell_lonlat[inverse], grid.centers[rows])


def test_values_are_normalised_with_the_statistics(paths, grid, tmp_path):
    groups = {
        "data": {"variables": ["LST"], "mean": [1000.0], "std": [10.0], "min": [0.0], "max": [0.0]},
        "data_daily": {"variables": ["dis24"], "mean": [1009.0], "std": [2.0], "min": [0.0], "max": [0.0]},
        "data_static": {"variables": ["DEM"], "mean": [1000.0], "std": [0.5], "min": [0.0], "max": [0.0]},
    }
    stats_zarr = tmp_path / "synthetic_stats.zarr"
    write_statistics(stats_zarr, groups)
    raw, _, _ = preload_modality(SOURCE, "synthetic", [19], paths, anchor_grid=grid, day_range=FIVE_DAYS)
    normalised, _, _ = preload_modality(
        SOURCE, "synthetic", [19], paths, stats_zarr, anchor_grid=grid, day_range=FIVE_DAYS
    )

    assert torch.allclose(normalised["data"], (raw["data"] - 1000.0) / 10.0, equal_nan=True)
    assert torch.allclose(normalised["data_daily"], (raw["data_daily"] - 1009.0) / 2.0)
    assert torch.allclose(normalised["data_static"], (raw["data_static"] - 1000.0) / 0.5)
    assert torch.allclose(normalised["metadata"], raw["metadata"], rtol=0.0, atol=0.0, equal_nan=True)  # kept as stored
    assert torch.equal(normalised["data_static_categorical"], raw["data_static_categorical"])


def test_the_transform_applies_to_the_data_leaf_only(paths, grid, tmp_path):
    groups = {
        "data": {"variables": ["LST"], "mean": [0.0], "std": [1.0], "min": [0.0], "max": [0.0]},
        "data_daily": {"variables": ["dis24"], "mean": [0.0], "std": [1.0], "min": [0.0], "max": [0.0]},
        "data_static": {"variables": ["DEM"], "mean": [0.0], "std": [1.0], "min": [0.0], "max": [0.0]},
    }
    stats_zarr = tmp_path / "synthetic_stats.zarr"
    write_statistics(stats_zarr, groups, transform="signed_log1p")
    source = dict(SOURCE, transform="signed_log1p")
    raw, _, _ = preload_modality(SOURCE, "synthetic", [19], paths, anchor_grid=grid, day_range=FIVE_DAYS)
    transformed, _, _ = preload_modality(
        source, "synthetic", [19], paths, stats_zarr, anchor_grid=grid, day_range=FIVE_DAYS
    )
    assert torch.allclose(transformed["data"], torch.log1p(raw["data"]), equal_nan=True)
    assert torch.equal(transformed["data_daily"], raw["data_daily"])
    assert torch.equal(transformed["data_static"], raw["data_static"])


def test_result_does_not_depend_on_blocks_or_threads(paths, grid, monkeypatch):
    anchor_ids = np.arange(0, 48, 5)
    ten_days = (FIRST_DAY - 1, FIRST_DAY + 9)
    one_block, _, _ = preload_modality(
        SOURCE, "synthetic", anchor_ids, paths, anchor_grid=grid, day_range=ten_days, num_workers=1
    )
    monkeypatch.setattr(preload, "SHARDS_PER_BLOCK", 1)
    many_blocks, _, _ = preload_modality(
        SOURCE, "synthetic", anchor_ids, paths, anchor_grid=grid, day_range=ten_days, num_workers=4
    )
    for leaf in SOURCE["load"]:
        assert torch.equal(
            torch.nan_to_num(one_block[leaf].float(), nan=-9.0), torch.nan_to_num(many_blocks[leaf].float(), nan=-9.0)
        )
    assert torch.isnan(one_block["data"][:, :24]).all()  # the day before the record
    assert not torch.isnan(one_block["data_daily"][0, 24:]).any()


def test_every_loaded_variable_needs_exactly_one_store():
    assert variable_stores(SOURCE)["quality_flag"] == "synthetic_hourly"
    in_no_store = {"stores": {"synthetic_hourly": ["LST"]}, "load": {"data": ["LST", "MLE"]}}
    with pytest.raises(ValueError, match="data.MLE"):
        variable_stores(in_no_store)
    in_two_stores = {"stores": {"a": ["quality_flag"], "b": ["quality_flag"]}, "load": {"metadata": ["quality_flag"]}}
    with pytest.raises(ValueError, match="exactly one store"):
        variable_stores(in_two_stores)


def test_a_static_store_cannot_feed_a_leaf_on_the_hourly_axis(paths, grid):
    source = {"stores": {"synthetic_static": ["DEM"]}, "load": {"data": ["DEM"]}}
    with pytest.raises(ValueError, match="cannot take"):
        preload_modality(source, "synthetic", [3], paths, anchor_grid=grid, day_range=FIVE_DAYS)


def test_stores_of_a_modality_must_share_a_grid(paths, grid, tmp_path, monkeypatch, product_store_writer):
    class OtherGridReader(CubeReader):
        MODALITY_ID = "other_grid"

    monkeypatch.setitem(stores.READERS, "other_grid", OtherGridReader)
    dtype = np.dtype([("slope", "<i2")])
    values = np.zeros((1, 4, 4), dtype=dtype)
    lat, lon = 5.0 - np.arange(4), 30.0 + np.arange(4)
    product_store_writer(tmp_path / "other_grid.zarr", "other_grid", None, lat, lon, 1, np.array([0]), values, (-1,))
    source = {
        "stores": {"synthetic_static": ["DEM"], "other_grid": ["slope"]},
        "load": {"data_static": ["DEM", "slope"]},
    }
    with pytest.raises(ValueError, match="same grid"):
        preload_modality(source, "synthetic", [3], paths, anchor_grid=grid, day_range=FIVE_DAYS)


def test_preload_modalities_returns_the_parts_of_the_shared_preload(paths, tmp_path):
    groups = {
        "data": {"variables": ["LST"], "mean": [1000.0], "std": [10.0], "min": [0.0], "max": [0.0]},
        "data_daily": {"variables": ["dis24"], "mean": [1009.0], "std": [2.0], "min": [0.0], "max": [0.0]},
        "data_static": {"variables": ["DEM"], "mean": [1000.0], "std": [0.5], "min": [0.0], "max": [0.0]},
    }
    write_statistics(paths.stats_zarr_path("ground", "tiny"), groups)
    sources = {"ground": SOURCE, "daily_only": {"stores": {"synthetic_daily": ["swir"]}, "load": {"data": ["swir"]}}}
    anchor_ids = np.array([3, 19])

    raw = preload_modalities(sources, ["ground", "daily_only"], "ground", anchor_ids, paths, day_range=FIVE_DAYS)
    assert set(raw) == {"columns", "inverse", "var_names", "cell_lonlat"}
    assert raw["var_names"]["ground"] == SOURCE["load"]
    assert raw["var_names"]["daily_only"] == {"data": ["swir"]}
    assert raw["columns"]["daily_only"]["data"].shape == (2, 120, 1)  # a daily store on the hourly axis
    assert raw["columns"]["daily_only"]["data"][1, 0, 0] == 2000 + 19 % 10
    assert raw["inverse"]["ground"].tolist() == [0, 1]

    normalised = preload_modalities(
        sources, ["ground"], "ground", anchor_ids, paths, stats_spatial="tiny", day_range=FIVE_DAYS
    )
    expected = (raw["columns"]["ground"]["data_static"] - 1000.0) / 0.5
    assert torch.allclose(normalised["columns"]["ground"]["data_static"], expected)
