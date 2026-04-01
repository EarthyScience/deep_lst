"""Tests for save_statistics_npz (pure numpy, no eodata required).

_load_indices tests live in test_compute_statistics.py because compute_statistics.py
has top-level eodata imports; those tests are skipped when eodata is not installed.
"""

import numpy as np
import pytest

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
