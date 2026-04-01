"""Tests for statistics I/O and index assembly.

Covers:
  - save_statistics_npz: round-trip correctness, suffix auto-correction, missing-key error
  - _load_indices: output shape, value correctness, fallback key 'mapping', missing key error
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# _load_indices lives in a script, not a package — add scripts dir to path.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "statistics"))
from compute_statistics import _load_indices

from deep_lst.utils.io import save_statistics_npz

# ── save_statistics_npz ───────────────────────────────────────────────────────


def test_save_statistics_npz_roundtrip(tmp_path):
    stats = {
        "t2m": {"mean": 280.5, "std": 15.2, "min": 210.0, "max": 320.0},
        "u10": {"mean": 0.1, "std": 5.0, "min": -30.0, "max": 30.0},
    }
    path = tmp_path / "statistics_test.npz"
    save_statistics_npz(stats, path)

    loaded = np.load(path, allow_pickle=True)
    variables = loaded["variables"].tolist()

    assert set(variables) == {"t2m", "u10"}
    for stat in ("mean", "std", "min", "max"):
        assert stat in loaded
        for var in ("t2m", "u10"):
            idx = variables.index(var)
            assert np.isclose(loaded[stat][idx], stats[var][stat])


def test_save_statistics_npz_suffix_autocorrect(tmp_path):
    stats = {"skt": {"mean": 290.0, "std": 10.0, "min": 250.0, "max": 330.0}}
    path_no_suffix = tmp_path / "statistics_test"  # no .npz
    save_statistics_npz(stats, path_no_suffix)
    assert (tmp_path / "statistics_test.npz").exists()


def test_save_statistics_npz_missing_stat_raises(tmp_path):
    stats = {
        "t2m": {"mean": 280.0, "std": 10.0},
        "u10": {"mean": 0.0},  # missing 'std'
    }
    with pytest.raises(KeyError, match="std"):
        save_statistics_npz(stats, tmp_path / "bad.npz")


# ── _load_indices ─────────────────────────────────────────────────────────────


def _write_spatial(tmp_path, key, values):
    path = tmp_path / "spatial.npz"
    np.savez(path, **{key: np.array(values, dtype=np.int64)})
    return path


def _write_temporal(tmp_path, values):
    path = tmp_path / "temporal.npz"
    np.savez(path, time_indices=np.array(values, dtype=np.int64))
    return path


def test_load_indices_shape(tmp_path):
    spatial = _write_spatial(tmp_path, "seviri_indices", [10, 20, 30])
    temporal = _write_temporal(tmp_path, [100, 200])

    indices = _load_indices("seviri", spatial, temporal)

    assert indices.shape == (6, 2)  # 3 spatial × 2 temporal


def test_load_indices_values(tmp_path):
    spatial = _write_spatial(tmp_path, "era5_indices", [5, 7])
    temporal = _write_temporal(tmp_path, [1, 2, 3])

    indices = _load_indices("era5", spatial, temporal)

    # meshgrid indexing='ij': spatial varies slowest
    assert indices.shape == (6, 2)
    spatial_col = indices[:, 0]
    temporal_col = indices[:, 1]
    assert set(spatial_col.tolist()) == {5, 7}
    assert set(temporal_col.tolist()) == {1, 2, 3}
    # each spatial index appears once per timestep
    assert (spatial_col == 5).sum() == 3
    assert (spatial_col == 7).sum() == 3


def test_load_indices_fallback_mapping_key(tmp_path):
    spatial = _write_spatial(tmp_path, "mapping", [0, 1, 2, 3])
    temporal = _write_temporal(tmp_path, [10])

    indices = _load_indices("seviri", spatial, temporal)

    assert indices.shape == (4, 2)


def test_load_indices_missing_key_raises(tmp_path):
    path = tmp_path / "bad_spatial.npz"
    np.savez(path, wrong_key=np.array([1, 2, 3]))
    temporal = _write_temporal(tmp_path, [1])

    with pytest.raises(KeyError, match="mapping"):
        _load_indices("seviri", path, temporal)
