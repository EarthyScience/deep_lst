"""Training path wiring, with tiny inputs instead of the institutional stores."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

pytest.importorskip("eodata")
from deep_lst import train  # noqa: E402
from deep_lst.dataset import dataset  # noqa: E402
from deep_lst.dataset.data_module import LitDataModule  # noqa: E402
from deep_lst.utils.io import ExperimentPaths  # noqa: E402

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"eozarr_root": "local-data"},
        {"preload_cache_root": "local-cache"},
        {"eozarr_root": "local-data", "preload_cache_root": "local-cache"},
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
def test_configured_reader_and_cache_preserve_samples_on_cold_and_warm_load(tmp_path, monkeypatch, explicit_paths):
    if explicit_paths:
        monkeypatch.chdir(tmp_path)
    root = Path(".") if explicit_paths else tmp_path
    paths = ExperimentPaths(assets_root=root / "assets", eozarr_root=root / "data", preload_cache_root=root / "cache")
    spatial = paths.spatial_indices_path("tiny")
    spatial.parent.mkdir(parents=True)
    np.savez(spatial, src_indices=[0])
    temporal = paths.temporal_indices_path("tiny")
    temporal.parent.mkdir(parents=True)
    np.savez(temporal, time_indices=[2, 3])
    stats = paths.stats_zarr_path("seviri", "mtg_new")
    stats.mkdir(parents=True)
    (stats / "zarr.json").write_text("{}")
    if not explicit_paths:
        monkeypatch.setattr(dataset, "PATHS", paths)
    original_paths = dataset.PATHS
    grid = SimpleNamespace(centers=np.array([[20.0, 10.0]]))
    monkeypatch.setattr(dataset, "anchor_grid", lambda _: grid)
    monkeypatch.setattr(dataset, "store_grid", lambda _: grid)
    monkeypatch.setattr(dataset, "map_to_store_rows", lambda *args: np.array([0]))
    monkeypatch.setattr(dataset, "get_data_reader_class", lambda _: lambda store, average_hourly: None)
    stores, normalizers, computed = [], [], []

    def reader(modality, store, **kwargs):
        stores.append(store)
        return SimpleNamespace(schema={}, data_arr=np.empty(0, dtype=[("LST", "f4")]))

    def normalizer(spec, schema):
        normalizers.append(spec)

    def computed_stats(path):
        computed.append(path)
        return {"solar_zenith": (0.0, 1.0)}

    column = torch.arange(48, dtype=torch.float32).reshape(1, 48, 1)
    column[0, 3] = float("nan")
    monkeypatch.setattr(dataset.eodata, "get_data_reader", reader)
    monkeypatch.setattr(dataset.ModalityNormalizer, "from_spec", staticmethod(normalizer))
    monkeypatch.setattr(dataset, "_preload_columns", lambda *args, **kwargs: {"data": column})
    monkeypatch.setattr(dataset, "load_computed_stats", computed_stats)
    split = dict(
        spatial="tiny",
        temporal="tiny",
        stats_spatial="mtg_new",
        seviri=dict(store="tiny", load={"data": ["LST"]}, lookback=2, horizon=2, prognostics=["LST"]),
    )
    for _ in range(2):
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
            torch.testing.assert_close(data[0]["seviri", "LST"].flatten(), column[0, :5, 0], equal_nan=True)
    assert stores == [str(paths.eozarr_store_path("tiny"))]
    assert normalizers == [f"zscore:{stats}"]
    assert computed == [stats] * 6
    assert len(list(paths.preload_cache_root.glob("preload_tiny_*.pt"))) == 1
    assert dataset.PATHS is original_paths
