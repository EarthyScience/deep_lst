"""Tests for _load_indices from scripts/statistics/compute_statistics.py.

Skipped automatically when eodata is not installed (CI without local zarr deps).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("eodata", reason="eodata not installed — skipping compute_statistics tests")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "statistics"))
from compute_statistics import _load_indices


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

    spatial_col = indices[:, 0]
    temporal_col = indices[:, 1]
    assert indices.shape == (6, 2)
    assert set(spatial_col.tolist()) == {5, 7}
    assert set(temporal_col.tolist()) == {1, 2, 3}
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
