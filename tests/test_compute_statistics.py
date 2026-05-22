"""Tests for load_indices from scripts/statistics/compute_statistics.py.

Skipped automatically when eodata is not installed (CI without local zarr deps).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("eodata", reason="eodata not installed — skipping compute_statistics tests")
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "statistics"))


def _write_spatial(tmp_path, mapping, src_indices):
    path = tmp_path / "spatial.npz"
    np.savez(path, mapping=np.asarray(mapping, dtype=np.int64), src_indices=np.asarray(src_indices, dtype=np.int64))
    return path


def _write_temporal(tmp_path, values):
    path = tmp_path / "temporal.npz"
    np.savez(path, time_indices=np.array(values, dtype=np.int64))
    return path
