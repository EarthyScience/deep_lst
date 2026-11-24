"""Training path wiring, with tiny inputs instead of the institutional stores."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst import stores, train
from deep_lst.dataset import dataset
from deep_lst.dataset.data_module import LitDataModule
from deep_lst.stores import T1H, CubeReader
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.statistics import write_statistics

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


class TinyLSTReader(CubeReader):
    """Reader of the synthetic hourly store of tests/conftest.py."""

    MODALITY_ID = "synthetic_hourly"
    TIME_GRID = T1H


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"product_root": "local-data"},
        {"preload_cache_root": "local-cache"},
        {"product_root": "local-data", "preload_cache_root": "local-cache"},
    ],
)
def test_training_passes_paths_to_data_module_and_saves_overrides(tmp_path, monkeypatch, overrides):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(
            config_name="train", overrides=["wandb_logger=false", *[f"paths.{k}={v}" for k, v in overrides.items()]]
        )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEP_LST_CACHE_ROOT", str(tmp_path / "environment-cache"))
    choices = dict(model="mlp", objective="rollout", loss="smoothl1", horizon="h24", data="full")
    monkeypatch.setattr(train.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(choices=choices)))
    monkeypatch.setattr(train, "check_stats_transform", lambda *args: None)
    monkeypatch.setattr(train.pl, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.torch, "set_float32_matmul_precision", lambda *args: None)
    received = []

    class DataModuleReached(Exception):
        pass

    def stop_at_data_module(**kwargs):
        received.append(kwargs.get("experiment_paths"))
        raise DataModuleReached

    monkeypatch.setattr(train, "LitDataModule", stop_at_data_module)
    with pytest.raises(DataModuleReached):
        train.main.__wrapped__(cfg)
    assert received == [ExperimentPaths(**{k: Path(v) for k, v in overrides.items()})]
    saved = OmegaConf.load(next(tmp_path.glob("experiments/*/version_*/config.yml")))
    assert saved.paths == cfg.paths
    assert not (tmp_path / "local-data").exists()


@pytest.mark.parametrize("explicit_paths", [False, True])
def test_configured_stores_and_cache_preserve_samples_on_cold_and_warm_load(
    tmp_path, monkeypatch, explicit_paths, hourly_store, product_store_writer
):
    """The data module reads a store through the configured paths, caches the preload, serves the
    same samples from the cache, and reads a store again when it gets a root of its own."""
    monkeypatch.setitem(stores.READERS, "synthetic_hourly", TinyLSTReader)
    paths = ExperimentPaths(
        assets_root=tmp_path / "assets", product_root=tmp_path, preload_cache_root=tmp_path / "cache"
    )
    anchor_row = 19  # the anchors are rows of the grid of the modality seviri
    spatial = paths.spatial_indices_path("tiny")
    spatial.parent.mkdir(parents=True)
    np.savez(spatial, src_indices=[anchor_row])
    temporal = paths.temporal_indices_path("tiny")
    temporal.parent.mkdir(parents=True)
    np.savez(temporal, time_indices=[2, 3])
    plain = {"mean": [0.0], "std": [1.0], "min": [0.0], "max": [0.0]}
    groups = {"data": dict(plain, variables=["LST"]), "computed": dict(plain, variables=["solar_zenith"])}
    write_statistics(paths.stats_zarr_path("seviri", "mtg_new"), groups)
    if not explicit_paths:
        monkeypatch.setattr(dataset, "PATHS", paths)
    original_paths = dataset.PATHS

    split = dict(
        spatial="tiny",
        anchor="seviri",
        temporal="tiny",
        stats_spatial="mtg_new",
        seviri=dict(
            stores={"synthetic_hourly": ["LST"]}, load={"data": ["LST"]}, lookback=2, horizon=2, prognostics=["LST"]
        ),
    )
    # LST of the anchor cell in the first five hours of 2016: 1000 + 10 * hour + row % 10
    expected = torch.tensor([1009.0, 1019.0, 1029.0, 1039.0, 1049.0])
    for _ in range(2):  # cold, then from the cache
        module = LitDataModule(
            experiment_paths=paths if explicit_paths else None,
            modality_ids=["seviri"],
            model={"name": "mlp"},
            train_dataset=split,
            val_dataset=split,
            test_dataset=split,
        )
        module.setup()
        for data in (module.train_dataset, module.val_dataset, module.test_dataset):
            np.testing.assert_array_equal(data.indices[:, 1], dataset.STORE_EPOCH_OFFSET + np.array([2, 3]))
            torch.testing.assert_close(data[0]["seviri", "LST"].flatten(), expected)
    assert len(list(paths.preload_cache_root.glob("preload_tiny_*.pt"))) == 1
    assert dataset.PATHS is original_paths

    if explicit_paths:  # the same store under a root of its own: read again, cached under another key
        changed_values = hourly_store.values.copy()
        changed_values["LST"] += 100
        changed_root = tmp_path / "changed-data"
        product_store_writer(
            changed_root / "synthetic_hourly.zarr",
            "synthetic_hourly",
            T1H,
            hourly_store.lat,
            hourly_store.lon,
            hourly_store.steps_per_shard,
            hourly_store.written_steps,
            changed_values,
            hourly_store.fill_value,
        )
        paths.store_roots["synthetic_hourly"] = str(changed_root)
        changed = LitDataModule(experiment_paths=paths, **module.kwargs)
        changed.setup()
        torch.testing.assert_close(changed.train_dataset[0]["seviri", "LST"].flatten(), expected + 100)
        assert len(list(paths.preload_cache_root.glob("preload_tiny_*.pt"))) == 2
