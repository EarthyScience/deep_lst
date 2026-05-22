"""Tests for UnimodalDataset and MultimodalDataset.

Uses a lightweight mock reader — no zarr stores, no real data.
The mock records every get_data_single call so tests can assert
that the correct (spatial_idx, time_idx) pairs are forwarded.
"""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("eodata", reason="eodata not installed — skipping dataset tests")
from deep_lst.dataset.dataset import MultimodalDataset, UnimodalDataset, load_samples


class MockReader:
    """Minimal reader stub. Returns a TensorDict with a fixed-shape tensor."""

    def __init__(self, out_shape=(4,)):
        self.calls: list[tuple[int, int]] = []
        self.out_shape = out_shape

    def get_data_single(self, spatial_idx: int, time_idx: int) -> TensorDict:
        self.calls.append((int(spatial_idx), int(time_idx)))
        return TensorDict(
            {"x": torch.zeros(self.out_shape)},
            batch_size=[],
        )


def _write_spatial(tmp_path, mapping, src_indices):
    path = tmp_path / "spatial.npz"
    np.savez(path, mapping=np.asarray(mapping, dtype=np.int64), src_indices=np.asarray(src_indices, dtype=np.int64))
    return path


def _write_temporal(tmp_path, values):
    path = tmp_path / "temporal.npz"
    np.savez(path, time_indices=np.array(values, dtype=np.int64))
    return path


def make_indices(n_spatial: int, n_time: int) -> np.ndarray:
    """Mesh spatial [0..n_spatial) × time [0..n_time), return (N,2) array."""
    space = np.arange(n_spatial, dtype=np.int64)
    time = np.arange(n_time, dtype=np.int64)
    s, t = np.meshgrid(space, time, indexing="ij")
    return np.stack([s.ravel(), t.ravel()], axis=1)


def test_load_samples_sorted(tmp_path):
    mapping_sorted = src_indices = [10, 20, 40, 50]
    spatial = _write_spatial(tmp_path, mapping_sorted, src_indices)
    temporal = _write_temporal(tmp_path, [0, 1, 2])  # single time index for simplicity

    samples = load_samples(spatial, temporal_indices_path=temporal)

    # 4 pixels × 3 timesteps, space-first meshgrid
    assert samples.shape == (4 * 3, 2)
    pixel_indices = samples[:, 0].reshape(4, 3)[:, 0]
    assert pixel_indices.tolist() == mapping_sorted


def test_load_samples_unsorted(tmp_path):
    mapping_unsorted = src_indices = [50, 10, 40, 20]
    sorted_mapping = [10, 20, 40, 50]
    spatial = _write_spatial(tmp_path, mapping_unsorted, src_indices)
    temporal = _write_temporal(tmp_path, [0, 1, 2])  # single time index for simplicity

    samples = load_samples(spatial, temporal_indices_path=temporal)

    # 4 pixels × 3 timesteps, space-first meshgrid
    assert samples.shape == (4 * 3, 2)
    pixel_indices = samples[:, 0].reshape(4, 3)[:, 0]
    assert pixel_indices.tolist() == sorted_mapping


def test_load_samples_crossmodal_alignment(tmp_path):
    spatial_sev = _write_spatial(tmp_path, [50, 10, 40, 20], [50, 10, 40, 20])
    spatial_era = _write_spatial(tmp_path, [500, 100, 400, 200], [50, 10, 40, 20])
    temporal = _write_temporal(tmp_path, [0, 1, 2])

    seviri_samples = load_samples(spatial_sev, temporal_indices_path=temporal)
    era5_samples = load_samples(spatial_era, temporal_indices_path=temporal)

    sev_src_sorted = np.sort(np.load(spatial_sev)["src_indices"])
    era_src_sorted = np.sort(np.load(spatial_era)["src_indices"])

    assert np.array_equal(sev_src_sorted, era_src_sorted)
    assert np.array_equal(seviri_samples[:, 0], era5_samples[:, 0])


def test_load_indices_missing_key_raises(tmp_path):
    path = tmp_path / "bad_spatial.npz"
    np.savez(path, wrong_key=np.array([1, 2, 3]))
    temporal = _write_temporal(tmp_path, [1])
    with pytest.raises(KeyError, match="mapping"):
        load_samples(path, temporal_indices_path=temporal)


def test_unimodal_len():
    indices = make_indices(3, 4)
    ds = UnimodalDataset(reader=MockReader(), indices=indices)
    assert len(ds) == 12


def test_unimodal_getitem_returns_tensordict():
    indices = make_indices(2, 3)
    ds = UnimodalDataset(reader=MockReader(), indices=indices)
    sample = ds[0]
    assert isinstance(sample, TensorDict)
    assert "x" in sample


def test_unimodal_getitem_forwards_correct_indices():
    indices = make_indices(2, 2)  # [[0,0],[0,1],[1,0],[1,1]]
    reader = MockReader()
    ds = UnimodalDataset(reader=reader, indices=indices)

    _ = ds[0]
    _ = ds[3]

    assert reader.calls[0] == (int(indices[0, 0]), int(indices[0, 1]))
    assert reader.calls[1] == (int(indices[3, 0]), int(indices[3, 1]))


def test_unimodal_all_indices_visited():
    indices = make_indices(3, 5)
    reader = MockReader()
    ds = UnimodalDataset(reader=reader, indices=indices)

    for i in range(len(ds)):
        ds[i]

    expected = [(int(r[0]), int(r[1])) for r in indices]
    assert reader.calls == expected


def test_unimodal_single_sample():
    indices = np.array([[7, 42]], dtype=np.int64)
    reader = MockReader()
    ds = UnimodalDataset(reader=reader, indices=indices)

    assert len(ds) == 1
    ds[0]
    assert reader.calls == [(7, 42)]


def test_multimodal_len():
    indices = {
        "era5": make_indices(3, 4),
        "seviri": make_indices(3, 4),
    }
    readers = {"era5": MockReader(), "seviri": MockReader()}
    ds = MultimodalDataset(reader=readers, indices=indices)
    assert len(ds) == 12


def test_multimodal_getitem_returns_tensordict_with_all_modalities():
    indices = {
        "era5": make_indices(2, 2),
        "seviri": make_indices(2, 2),
    }
    readers = {"era5": MockReader(), "seviri": MockReader()}
    ds = MultimodalDataset(reader=readers, indices=indices)

    sample = ds[0]
    assert isinstance(sample, TensorDict)
    assert "era5" in sample
    assert "seviri" in sample


def test_multimodal_each_reader_called_with_its_own_indices():
    era5_indices = make_indices(2, 2)  # [[0,0],[0,1],[1,0],[1,1]]
    seviri_indices = make_indices(2, 2)

    era5_reader = MockReader(out_shape=(4,))
    seviri_reader = MockReader(out_shape=(8,))

    ds = MultimodalDataset(
        reader={"era5": era5_reader, "seviri": seviri_reader},
        indices={"era5": era5_indices, "seviri": seviri_indices},
    )

    ds[2]  # third row: spatial=1, time=0

    assert era5_reader.calls[-1] == (int(era5_indices[2, 0]), int(era5_indices[2, 1]))
    assert seviri_reader.calls[-1] == (int(seviri_indices[2, 0]), int(seviri_indices[2, 1]))


def test_multimodal_single_modality():
    indices = {"era5": make_indices(4, 1)}
    readers = {"era5": MockReader()}
    ds = MultimodalDataset(reader=readers, indices=indices)

    assert len(ds) == 4
    sample = ds[0]
    assert "era5" in sample
