"""The statistics script and the baselines with the configured stores and store roots."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from deep_lst import stores
from deep_lst.baselines.era5land import Era5LandExtractor
from deep_lst.stores import T1H, CubeReader
from deep_lst.utils.io import ExperimentPaths, load_computed_stats, load_inc_variances
from deep_lst.utils.statistics import read_variable_statistics
from scripts.assets import compute_statistics


class TinyLSTReader(CubeReader):
    """Reader of the synthetic hourly store of tests/conftest.py."""

    MODALITY_ID = "synthetic_hourly"
    TIME_GRID = T1H


@pytest.mark.parametrize("own_root", [False, True])
def test_statistics_command_reads_the_store_under_its_root(
    tmp_path, monkeypatch, own_root, hourly_store, product_store_writer
):
    """compute_statistics.py end to end on a synthetic store: two cells, the first two days of
    2016, with the store under product_root or under a root of its own."""
    monkeypatch.chdir(tmp_path)  # the script writes below ./assets
    monkeypatch.setitem(stores.READERS, "synthetic_hourly", TinyLSTReader)
    paths = ExperimentPaths()
    rows = [3, 19]
    spatial = paths.spatial_indices_path("tiny")
    spatial.parent.mkdir(parents=True)
    np.savez(spatial, src_indices=rows)
    temporal = paths.temporal_indices_path("tiny")
    temporal.parent.mkdir(parents=True)
    np.savez(temporal, time_indices=[2, 3, 26])  # hours of 2016-01-01 and 2016-01-02

    offset = 0
    store_roots = {}
    if own_root:  # a second copy with other values; the script must read this one
        offset = 100
        other_values = hourly_store.values.copy()
        other_values["LST"] += offset
        other_root = tmp_path / "other-root"
        product_store_writer(
            other_root / "synthetic_hourly.zarr",
            "synthetic_hourly",
            T1H,
            hourly_store.lat,
            hourly_store.lon,
            hourly_store.steps_per_shard,
            hourly_store.written_steps,
            other_values,
            hourly_store.fill_value,
        )
        store_roots = {"synthetic_hourly": str(other_root)}

    source = {"stores": {"synthetic_hourly": ["LST"]}, "load": {"data": ["LST"]}, "increment_vars": ["LST"]}
    cfg = OmegaConf.create(
        dict(
            modality="seviri",
            sources={"seviri": source},
            spatial="tiny",
            anchor="seviri",
            temporal="tiny",
            num_workers=2,
            paths={"product_root": str(tmp_path), "store_roots": store_roots},
        )
    )
    compute_statistics.main.__wrapped__(cfg)

    # the two cells over the 48 hours of the two days; hours 30 to 32 were never written
    hours = np.arange(48)
    lst = np.full((2, 48), np.nan)
    written = (hours < 30) | (hours > 32)
    positions = np.where(hours < 30, hours, hours - 3)[written]
    for index, row in enumerate(rows):
        lst[index, written] = 1000 + (positions % 50) * 10 + row % 10 + offset
    stats_zarr = paths.stats_zarr_path("seviri", "tiny")
    mean, std = read_variable_statistics(stats_zarr)["LST"]
    assert mean == pytest.approx(np.nanmean(lst))
    assert std == pytest.approx(np.nanstd(lst, ddof=1))
    increments = np.diff(lst.reshape(2, 2, 24), axis=2)  # hourly changes within each day
    increment_std = np.nanstd(increments, ddof=1)
    assert load_inc_variances(stats_zarr) == {"LST": pytest.approx((increment_std / std) ** 2)}
    assert set(load_computed_stats(stats_zarr)) == {"solar_zenith"}


def test_baseline_attributes_come_from_the_variable_table():
    extractor = Era5LandExtractor("skt", 2024)
    assert extractor._metadata_attrs("skt") == dict(
        variable="skt", units="K", long_name="Skin temperature", test_year="2024"
    )
    assert extractor._metadata_attrs("unlisted") == dict(
        variable="unlisted", units="", long_name="unlisted", test_year="2024"
    )
