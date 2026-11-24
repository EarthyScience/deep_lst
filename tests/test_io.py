"""Tests for ExperimentPaths path resolution.

All path-structure tests use a custom root via tmp_path so they are
hermetic — nothing depends on the real on-disk layout.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from deep_lst.utils.io import ExperimentPaths

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"


@pytest.mark.parametrize("entrypoint", ["train", "finetune", "infer", "statistics", "sensitivity"])
def test_entrypoints_use_the_shared_paths_file(entrypoint):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        default = compose(config_name=entrypoint)
        cfg = compose(config_name=entrypoint, overrides=["+paths.store_roots.seviri=local-seviri"])
    assert ExperimentPaths.from_config(default.paths) == ExperimentPaths()
    paths = ExperimentPaths.from_config(cfg.paths)
    assert paths.store_roots == {"seviri": "local-seviri"}
    assert paths.eozarr_root == ExperimentPaths().eozarr_root


@pytest.fixture
def paths(tmp_path):
    return ExperimentPaths(
        experiment_root=tmp_path / "experiments",
        config_root=tmp_path / "configs",
        assets_root=tmp_path / "assets",
    )


def test_cache_root_uses_the_configured_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEEP_LST_CACHE_ROOT", str(tmp_path / "old-env-cache"))
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["paths.preload_cache_root=chosen-cache"])
    assert ExperimentPaths().preload_cache_root == Path("cache")
    paths = ExperimentPaths.from_config(cfg.paths)
    assert paths.preload_cache_path("fingerprint").resolve() == tmp_path / "chosen-cache/preload_fingerprint.pt"
    assert not (tmp_path / "old-env-cache").exists()


def test_legacy_cache_parent_uses_the_yaml_default():
    paths = ExperimentPaths.from_config({"preload_cache_root": None, "preload_cache_parent": "/old/scratch"})
    assert paths.preload_cache_root == Path("cache")


@pytest.mark.parametrize("root_kind", ["default", "string", "path", "relative"])
def test_store_root_override_keeps_directory_and_zip_selection(tmp_path, monkeypatch, root_kind):
    monkeypatch.chdir(tmp_path)
    paths = ExperimentPaths(eozarr_root=tmp_path / "shared")
    root = paths.eozarr_root if root_kind == "default" else tmp_path / "modality"
    if root_kind == "relative":
        root = Path("modality")
    kwargs = {} if root_kind == "default" else {"store_root": str(root) if root_kind == "string" else root}
    directory = root / "tiny.eozarr"
    archive = root / "tiny.eozarr.zip"
    root.mkdir(parents=True)
    assert paths.eozarr_store_path("tiny", **kwargs) == directory
    archive.touch()
    assert paths.eozarr_store_path("tiny", **kwargs) == archive
    (directory / "data").mkdir(parents=True)
    (directory / "data" / "zarr.json").write_text("{}")
    assert paths.eozarr_store_path("tiny", **kwargs) == directory
    assert paths.eozarr_root == tmp_path / "shared"


# ── directory resolution ──────────────────────────────────────────────────────


def test_experiment_dir(paths, tmp_path):
    assert paths.experiment_dir("run_01") == tmp_path / "experiments" / "run_01"


def test_checkpoint_dir(paths, tmp_path):
    assert paths.checkpoint_dir("run_01", "v0") == (tmp_path / "experiments" / "run_01" / "v0" / "checkpoints")


# ── file resolution ───────────────────────────────────────────────────────────


def test_config_path(paths, tmp_path):
    assert paths.config_path("mlp_diag") == (tmp_path / "configs" / "mlp_diag" / "config.yml")


def test_statistics_path(paths, tmp_path):
    assert paths.statistics_path("era5", "global") == (
        tmp_path / "assets" / "era5" / "statistics" / "statistics_global.npz"
    )


def test_spatial_indices_path(paths, tmp_path):
    assert paths.spatial_indices_path("mtg") == (tmp_path / "assets" / "spatial" / "indices_mtg.npz")


def test_temporal_indices_path(paths, tmp_path):
    assert paths.temporal_indices_path("train") == (tmp_path / "assets" / "temporal" / "time_indices_train.npz")


# ── latest_checkpoint ─────────────────────────────────────────────────────────


def test_latest_checkpoint_returns_last_alphabetically(paths):
    ckpt_dir = paths.checkpoint_dir("run_01", "v0")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch=00.ckpt").touch()
    (ckpt_dir / "epoch=01.ckpt").touch()
    (ckpt_dir / "epoch=02.ckpt").touch()

    assert paths.latest_checkpoint("run_01", "v0").name == "epoch=02.ckpt"


def test_latest_checkpoint_single_file(paths):
    ckpt_dir = paths.checkpoint_dir("run_01", "v0")
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best.ckpt").touch()

    assert paths.latest_checkpoint("run_01", "v0").name == "best.ckpt"


def test_latest_checkpoint_empty_dir_raises(paths):
    ckpt_dir = paths.checkpoint_dir("run_01", "v0")
    ckpt_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        paths.latest_checkpoint("run_01", "v0")


def test_latest_checkpoint_missing_dir_raises(paths):
    with pytest.raises(FileNotFoundError):
        paths.latest_checkpoint("nonexistent", "v0")
