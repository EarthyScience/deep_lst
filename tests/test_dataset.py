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


def _write_temporal(tmp_path, values):
    path = tmp_path / "temporal.npz"
    np.savez(path, time_indices=np.array(values, dtype=np.int64))
    return path


def make_indices(n_spatial: int, n_time: int) -> np.ndarray:
    """Time-major mesh: row i*n_spatial + j = (spatial j, time i), matching load_samples."""
    space = np.arange(n_spatial, dtype=np.int64)
    time = np.arange(n_time, dtype=np.int64)
    t, s = np.meshgrid(time, space, indexing="ij")
    return np.stack([s.ravel(), t.ravel()], axis=1)


def test_load_samples_meshgrid_layout(tmp_path):
    """load_samples returns (n_time * n_spatial, 2) in time-major order:
    indices[i*n_spatial + j] = (spatial_idx[j], time_idx[i])."""
    temporal = _write_temporal(tmp_path, [0, 1, 2])
    spatial_idx = np.array([10, 20, 40, 50], dtype=np.int64)

    samples = load_samples(spatial_idx, temporal_indices_path=str(temporal))

    assert samples.shape == (3 * 4, 2)
    # First n_spatial entries are time 0 across all pixels; then time 1; etc.
    assert (samples[0:4, 1] == 0).all()
    assert (samples[4:8, 1] == 1).all()
    assert (samples[8:12, 1] == 2).all()
    # Pixel column cycles through [10, 20, 40, 50] within each init_time block.
    assert (samples[:, 0].reshape(3, 4) == np.array([10, 20, 40, 50])).all()


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
    indices = make_indices(2, 2)  # time-major: [[0,0],[1,0],[0,1],[1,1]]
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


def _multimodal_simple(n_pixels: int, n_time: int):
    """Build minimal MultimodalDataset args for (n_pixels × n_time) samples with
    identity spatial_lookups (each modality's id == seviri position)."""
    indices = make_indices(n_pixels, n_time)
    identity = np.arange(n_pixels, dtype=np.int64)
    spatial_lookups = {"era5": identity, "seviri": identity}
    readers = {"era5": MockReader(), "seviri": MockReader()}
    return readers, indices, spatial_lookups, n_pixels


def test_multimodal_len():
    readers, indices, lookups, n_spatial = _multimodal_simple(3, 4)
    ds = MultimodalDataset(reader=readers, indices=indices, spatial_lookups=lookups, n_spatial=n_spatial)
    assert len(ds) == 12


def test_multimodal_getitem_returns_tensordict_with_all_modalities():
    readers, indices, lookups, n_spatial = _multimodal_simple(2, 2)
    ds = MultimodalDataset(reader=readers, indices=indices, spatial_lookups=lookups, n_spatial=n_spatial)

    sample = ds[0]
    assert isinstance(sample, TensorDict)
    assert "era5" in sample
    assert "seviri" in sample


def test_multimodal_readers_called_via_spatial_lookup():
    """Each modality's reader receives the spatial id from spatial_lookups[modality],
    keyed on the shared seviri position. This is the fix for the cardinality-mismatch
    bug: SEVIRI and ERA5 may live in different spatial id spaces, joined by per-position
    mapping rather than zipped by raw idx."""
    n_pixels, n_time = 3, 4
    indices = make_indices(n_pixels, n_time)
    spatial_lookups = {
        "seviri": np.arange(n_pixels, dtype=np.int64),
        "era5": np.array([0, 10, 20], dtype=np.int64),  # different space, position-aligned
    }
    era5_reader, seviri_reader = MockReader(), MockReader()
    ds = MultimodalDataset(
        reader={"era5": era5_reader, "seviri": seviri_reader},
        indices=indices,
        spatial_lookups=spatial_lookups,
        n_spatial=n_pixels,
    )

    # Time-major: idx=5 → seviri_pos = 5 % 3 = 2, time = 5 // 3 = 1
    # → seviri reader gets (2,1), era5 reader gets (lookup[2]=20, 1)
    ds[5]
    assert seviri_reader.calls[-1] == (2, 1)
    assert era5_reader.calls[-1] == (20, 1)


def test_multimodal_single_modality():
    indices = make_indices(4, 1)
    lookups = {"seviri": np.arange(4, dtype=np.int64)}
    readers = {"seviri": MockReader()}
    ds = MultimodalDataset(reader=readers, indices=indices, spatial_lookups=lookups, n_spatial=4)

    assert len(ds) == 4
    sample = ds[0]
    assert "seviri" in sample


def test_multimodal_time_idx_and_spatial_idx_properties():
    """Dataset exposes its true coverage via .time_idx and .spatial_idx — these are
    the single source of truth used by setup/inference to size the forecast.zarr."""
    readers, indices, lookups, n_spatial = _multimodal_simple(3, 4)
    ds = MultimodalDataset(reader=readers, indices=indices, spatial_lookups=lookups, n_spatial=n_spatial)

    assert np.array_equal(ds.spatial_idx, np.array([0, 1, 2]))
    assert np.array_equal(ds.time_idx, np.array([0, 1, 2, 3]))


def test_unimodal_time_idx_and_spatial_idx_properties():
    indices = make_indices(3, 4)
    ds = UnimodalDataset(reader=MockReader(), indices=indices)

    assert np.array_equal(ds.spatial_idx, np.array([0, 1, 2]))
    assert np.array_equal(ds.time_idx, np.array([0, 1, 2, 3]))
