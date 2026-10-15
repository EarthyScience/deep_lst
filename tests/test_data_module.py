"""LitDataModule.setup: one preload per distinct spatial selection, shared across its splits."""

import pytest

pytest.importorskip("eodata", reason="eodata not installed — skipping data module tests")
from deep_lst.dataset.data_module import LitDataModule, MultimodalDataset  # noqa: E402


def _split(spatial):
    return {"spatial": spatial, "seviri": {"prognostics": ["LST"], "inputs": ["DEM"], "context_vars": ["LST"]}}


def _config(train_spatial, val_spatial, test_spatial):
    return {
        "modality_ids": ["seviri"],
        "model": {"name": "transformer"},
        "train_dataset": _split(train_spatial),
        "val_dataset": _split(val_spatial),
        "test_dataset": _split(test_spatial),
    }


@pytest.fixture
def fake_preload(monkeypatch):
    preload_calls = []
    build_calls = []

    def preload_shared(modality_ids, **kwargs):
        preload_calls.append(kwargs["spatial"])
        return {"spatial": kwargs["spatial"]}

    def from_modalities(modality_ids, shared, **kwargs):
        build_calls.append((kwargs["spatial"], shared["spatial"]))
        return shared

    monkeypatch.setattr(MultimodalDataset, "preload_shared", staticmethod(preload_shared))
    monkeypatch.setattr(MultimodalDataset, "from_modalities", staticmethod(from_modalities))
    return preload_calls, build_calls


def test_one_preload_when_all_splits_share_the_spatial(fake_preload):
    preload_calls, build_calls = fake_preload
    LitDataModule(**_config("mtg_new", "mtg_new", "mtg_new")).setup()
    assert preload_calls == ["mtg_new"]
    assert build_calls == [("mtg_new", "mtg_new")] * 3


def test_second_preload_for_a_held_out_val_test_selection(fake_preload):
    preload_calls, build_calls = fake_preload
    LitDataModule(**_config("mtg_new", "eval_mtg_new_val", "eval_mtg_new_val")).setup()
    assert preload_calls == ["mtg_new", "eval_mtg_new_val"]
    assert build_calls == [
        ("mtg_new", "mtg_new"),
        ("eval_mtg_new_val", "eval_mtg_new_val"),
        ("eval_mtg_new_val", "eval_mtg_new_val"),
    ]
