"""Per-modality store roots for statistics, baselines and full-store readers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("eodata")
from deep_lst.baselines import base  # noqa: E402
from deep_lst.baselines.era5land import Era5LandExtractor  # noqa: E402
from deep_lst.dataset import dataset  # noqa: E402
from deep_lst.utils import full_store  # noqa: E402
from deep_lst.utils.io import ExperimentPaths  # noqa: E402
from scripts.assets import compute_statistics, run_baselines  # noqa: E402


def test_full_store_reads_indices_and_columns_with_the_same_paths(tmp_path, monkeypatch):
    paths = ExperimentPaths(assets_root=tmp_path / "assets", store_roots={"seviri": str(tmp_path / "seviri")})
    indices = paths.spatial_indices_path("tiny")
    indices.parent.mkdir(parents=True)
    np.savez(indices, src_indices=[0])
    monkeypatch.setattr(full_store, "anchor_grid", lambda _: None)
    monkeypatch.setattr(full_store, "store_grid", lambda _: None)
    monkeypatch.setattr(full_store, "map_to_store_rows", lambda *args: np.array([4]))
    monkeypatch.setattr(full_store, "get_data_reader_class", lambda _: lambda store, average_hourly: None)

    def open_reader(modality, store, **kwargs):
        assert store == str(tmp_path / "seviri/tiny.eozarr")
        return SimpleNamespace(data_arr=np.empty(0, dtype=[("LST", "f4")]))

    monkeypatch.setattr(full_store.eodata, "get_data_reader", open_reader)
    monkeypatch.setattr(full_store, "_preload_columns", lambda *args, **kwargs: {"data": np.array([[281.0]])})
    columns, rows, inverse, anchors = full_store.load_store_columns(
        "seviri", "tiny", "tiny", {"data": ["LST"]}, paths=paths
    )
    np.testing.assert_array_equal(columns["data"], [[281.0]])
    np.testing.assert_array_equal(rows, [4])
    np.testing.assert_array_equal(inverse, [0])
    np.testing.assert_array_equal(anchors, [0])


@pytest.mark.parametrize(
    "store_roots, legacy", [({}, None), ({"seviri": "custom-store"}, "old-store"), ({}, "old-store")]
)
def test_statistics_dataset_uses_the_selected_store_root(tmp_path, monkeypatch, store_roots, legacy):
    monkeypatch.chdir(tmp_path)
    paths = ExperimentPaths(
        assets_root=tmp_path / "assets", eozarr_root=tmp_path / "default-store", store_roots=store_roots
    )
    for path, values in (
        (paths.spatial_indices_path("tiny"), {"src_indices": [0]}),
        (paths.temporal_indices_path("tiny"), {"time_indices": [0, 24]}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
    monkeypatch.setattr(dataset, "anchor_grid", lambda _: None)
    monkeypatch.setattr(dataset, "store_grid", lambda _: None)
    monkeypatch.setattr(dataset, "map_to_store_rows", lambda *args: np.array([4]))
    monkeypatch.setattr(dataset, "get_data_reader_class", lambda _: lambda store, average_hourly: None)
    stores = []
    reader = SimpleNamespace()

    def open_reader(modality, store, **kwargs):
        stores.append(store)
        return reader

    monkeypatch.setattr(dataset.eodata, "get_data_reader", open_reader)
    result = dataset.UnimodalDataset.from_config(
        "seviri", "tiny", "tiny", "mtg", "tiny", store_root=legacy, paths=paths
    )
    store_root = store_roots.get("seviri", legacy)
    expected_root = paths.eozarr_root if store_root is None else Path(store_root)
    assert stores == [str(expected_root / "tiny.eozarr")]
    assert result.reader is reader
    np.testing.assert_array_equal(result.rows, [4])
    assert result.stat_days == [dataset.PRELOAD_START_DAY, dataset.PRELOAD_START_DAY + 1]


@pytest.mark.parametrize("store_root", [None, "custom-store"])
def test_statistics_command_forwards_the_paths(monkeypatch, store_root):
    cfg = OmegaConf.create(
        dict(
            modality="seviri",
            sources={"seviri": {"store": "tiny"}},
            spatial="tiny",
            anchor="mtg",
            temporal="tiny",
            paths={"eozarr_root": "shared", "store_roots": {"seviri": store_root}},
        )
    )

    def stop_at_dataset(**kwargs):
        assert kwargs["store"] == "tiny"
        assert kwargs["paths"].store_roots["seviri"] == store_root
        assert kwargs["paths"].eozarr_root == Path("shared")
        raise RuntimeError("statistics dataset reached")

    monkeypatch.setattr(compute_statistics.UnimodalDataset, "from_config", staticmethod(stop_at_dataset))
    with pytest.raises(RuntimeError, match="statistics dataset reached"):
        compute_statistics.main.__wrapped__(cfg)


@pytest.mark.parametrize("store_root", [None, "custom-store"])
def test_baseline_metadata_uses_the_modality_store_root(tmp_path, monkeypatch, store_root):
    monkeypatch.chdir(tmp_path)
    extractor = Era5LandExtractor("skt", 2024)
    extractor.paths = ExperimentPaths(eozarr_root=tmp_path / "default-store")
    extractor.paths.store_roots["era5_land"] = store_root
    extractor.config = {"test_dataset": {"era5_land": {"store": "tiny"}}}
    stores = []

    def open_reader(modality, store, **kwargs):
        assert modality == "era5_land"
        stores.append(store)
        return SimpleNamespace(data_arr=SimpleNamespace(attrs={"metadata": {"skt": {"units": "K"}}}))

    monkeypatch.setattr(base, "get_data_reader", open_reader)
    attrs = extractor._metadata_attrs("era5_land", "tiny", "skt")
    expected_root = extractor.paths.eozarr_root if store_root is None else Path(store_root)
    assert stores == [str(expected_root / "tiny.eozarr")]
    assert attrs == dict(variable="skt", units="K", long_name="skt", description="", test_year="2024")


def test_baselines_read_the_selected_paths_profile():
    config = run_baselines.build_config("gobabeb", "local")
    paths = ExperimentPaths.from_config(config["paths"])
    assert paths.eozarr_root == Path("data")
    assert paths.preload_cache_root == Path("cache/local")
    assert config["test_dataset"]["spatial"] == "eval_gobabeb"
