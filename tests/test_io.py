"""Tests for ExperimentPaths path resolution.

All path-structure tests use a custom root via tmp_path so they are
hermetic — nothing depends on the real on-disk layout.
"""

import pytest

from deep_lst.utils.io import ExperimentPaths


@pytest.fixture
def paths(tmp_path):
    return ExperimentPaths(
        experiment_root=tmp_path / "experiments",
        config_root=tmp_path / "configs",
        assets_root=tmp_path / "assets",
    )


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
