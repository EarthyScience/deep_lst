"""Per-modality store roots for statistics and baseline metadata readers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("eodata")
from deep_lst.baselines import base  # noqa: E402
from deep_lst.baselines.era5land import Era5LandExtractor  # noqa: E402
from deep_lst.dataset import dataset  # noqa: E402
from deep_lst.utils.io import ExperimentPaths  # noqa: E402
from scripts.assets import compute_statistics  # noqa: E402


@pytest.mark.parametrize("store_root", [None, "custom-store"])
def test_statistics_dataset_uses_the_selected_store_root(tmp_path, monkeypatch, store_root):
    monkeypatch.chdir(tmp_path)
    paths = ExperimentPaths(assets_root=tmp_path / "assets", eozarr_root=tmp_path / "default-store")
    for path, values in (
        (paths.spatial_indices_path("tiny"), {"src_indices": [0]}),
        (paths.temporal_indices_path("tiny"), {"time_indices": [0, 24]}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **values)
    monkeypatch.setattr(dataset, "PATHS", paths)
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
    kwargs = {} if store_root is None else {"store_root": store_root}
    result = dataset.UnimodalDataset.from_config("seviri", "tiny", "tiny", "mtg", "tiny", **kwargs)
    expected_root = paths.eozarr_root if store_root is None else Path(store_root)
    assert stores == [str(expected_root / "tiny.eozarr")]
    assert result.reader is reader
    np.testing.assert_array_equal(result.rows, [4])
    assert result.stat_days == [dataset.PRELOAD_START_DAY, dataset.PRELOAD_START_DAY + 1]


@pytest.mark.parametrize("store_root", [None, "custom-store"])
def test_statistics_command_forwards_the_source_store_root(monkeypatch, store_root):
    source = {"store": "tiny"}
    if store_root is not None:
        source["store_root"] = store_root
    cfg = OmegaConf.create(
        dict(modality="seviri", sources={"seviri": source}, spatial="tiny", anchor="mtg", temporal="tiny")
    )

    def stop_at_dataset(**kwargs):
        assert kwargs["store"] == "tiny"
        assert kwargs.get("store_root") == store_root
        raise RuntimeError("statistics dataset reached")

    monkeypatch.setattr(compute_statistics.UnimodalDataset, "from_config", staticmethod(stop_at_dataset))
    with pytest.raises(RuntimeError, match="statistics dataset reached"):
        compute_statistics.main.__wrapped__(cfg)


@pytest.mark.parametrize("store_root", [None, "custom-store"])
def test_baseline_metadata_uses_the_modality_store_root(tmp_path, monkeypatch, store_root):
    monkeypatch.chdir(tmp_path)
    extractor = Era5LandExtractor("skt", 2024)
    extractor.paths = ExperimentPaths(eozarr_root=tmp_path / "default-store")
    source = {"store": "tiny"}
    if store_root is not None:
        source["store_root"] = store_root
    extractor.config = {"test_dataset": {"era5_land": source}}
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
