"""Golden tests for the Hydra configuration refactor (objective / window / sources).

tests/fixtures/config_v1_<case>.yml hold what five legacy command lines composed at develop
5274122 (`train.py ... --cfg job --resolve`). They are the reference for "no behavioural
change" and are never regenerated.

  test_legacy_composition_matches_the_fixture
      conf/ still composes the fixtures. Passes before the refactor; drop it in the stage
      that removes the legacy option names (model=mlp_rollout, horizon=h24).

The remaining tests describe the refactored configuration and are strict xfails until it
exists (deep_lst.utils.config.split_views / migrate_config and the groups objective, optim,
window, sources, features, splits):

  test_split_views_reproduce_the_legacy_datasets
      the three per-split dicts the datasets consume are unchanged
  test_relocated_keys_keep_their_values
      every key that moved kept its value; RELOCATED is the list of allowed relocations
  test_architecture_keys_stay_in_model
      what is left under `model` is the architecture, unchanged
  test_migrated_fixture_equals_the_new_composition, test_migration_is_idempotent
      a legacy config.yml migrates to what the new command line composes
"""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.utils import config as config_utils

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
FIXTURE_DIR = Path(__file__).parent / "fixtures"

# case -> the command line that wrote the fixture
LEGACY_OVERRIDES = {
    "default": ["model=mlp_rollout", "loss=smoothl1", "horizon=h24", "data=full"],
    "transformer": ["model=transformer_rollout", "loss=smoothl1", "horizon=h24", "data=full"],
    "mlp_diag": ["model=mlp_diag", "loss=mse", "horizon=h24", "data=full_lstdiag"],
    "mlp_gated": ["model=mlp_gated", "loss=smoothl1", "horizon=h24", "data=full_joint"],
    "h72_noera5": ["model=mlp_rollout", "loss=smoothl1", "horizon=h72", "data=noera5"],
}

# case -> the equivalent command line after the refactor
NEW_OVERRIDES = {
    "default": ["model=mlp", "objective=rollout", "loss=smoothl1", "window=h24", "data=full"],
    "transformer": ["model=transformer", "objective=rollout", "loss=smoothl1", "window=h24", "data=full"],
    "mlp_diag": ["model=mlp", "objective=diagnostic", "loss=mse", "window=h24", "data=full_lstdiag"],
    "mlp_gated": ["model=mlp", "objective=rollout_gated", "loss=smoothl1", "window=h24", "data=full_joint"],
    "h72_noera5": ["model=mlp", "objective=rollout", "loss=smoothl1", "window=h72", "data=noera5"],
}

# legacy path -> new path; the value must be equal. A legacy path absent from a fixture is
# skipped (mlp_diag sets no rollout keys).
RELOCATED = {
    "model.learning_rate": "optim.learning_rate",
    "model.weight_decay": "optim.weight_decay",
    "model.warmup_steps": "optim.warmup_steps",
    "model.scheduler_T0": "optim.scheduler_T0",
    "model.detach_rollout": "objective.detach_rollout",
    "model.init_fill": "objective.init_fill",
    "model.init_hide_fraction": "objective.init_hide_fraction",
    "model.checkpoint_monitor": "objective.checkpoint_monitor",
    "train_dataset.seviri.lookback": "window.lookback",
    "train_dataset.seviri.horizon": "window.horizon",
}
# legacy boolean -> weight of the loss term (true = 1.0, false = 0.0)
RELOCATED_TERMS = {
    "model.teacher_forcing_loss": "objective.terms.teacher_forcing",
    "model.incremental_loss": "objective.terms.increment",
    "model.rollout_loss": "objective.terms.state",
}
OBJECTIVE_TARGETS = {
    "RolloutLoss": "deep_lst.losses.RolloutLoss",
    "DiagnosticMSE": "deep_lst.losses.DiagnosticLoss",
}
POINTWISE_TARGETS = {
    "smooth_l1": "torch.nn.functional.smooth_l1_loss",
    "mse": "torch.nn.functional.mse_loss",
    "mae": "torch.nn.functional.l1_loss",
}
# keys that leave `model`; everything else under `model` is architecture and stays
MOVED_OUT_OF_MODEL = [
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "scheduler_T0",
    "detach_rollout",
    "init_fill",
    "init_hide_fraction",
    "checkpoint_monitor",
    "teacher_forcing_loss",
    "incremental_loss",
    "rollout_loss",
    "step_loss",
    "loss_fn",
    "probabilistic",
]
# the sections migrate_config has to rebuild exactly; naming tokens and runtime flags are not compared
MIGRATED_SECTIONS = [
    "config_version",
    "modality_ids",
    "model",
    "objective",
    "optim",
    "window",
    "sources",
    "features",
    "splits",
    "training",
]

needs_refactored_config = pytest.mark.xfail(
    strict=True,
    reason="needs split_views / migrate_config and the objective, optim, window, sources, features groups",
)


def _fixture(case):
    return yaml.safe_load((FIXTURE_DIR / f"config_v1_{case}.yml").read_text())


def _compose(overrides):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _has(config, dotted_path):
    node = config
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _lookup(config, dotted_path):
    node = config
    for key in dotted_path.split("."):
        node = node[key]
    return node


@pytest.mark.parametrize("case", list(LEGACY_OVERRIDES))
def test_legacy_composition_matches_the_fixture(case):
    assert _compose(LEGACY_OVERRIDES[case]) == _fixture(case)


@needs_refactored_config
@pytest.mark.parametrize("case", list(NEW_OVERRIDES))
def test_split_views_reproduce_the_legacy_datasets(case):
    legacy = _fixture(case)

    views = config_utils.split_views(_compose(NEW_OVERRIDES[case]))

    assert views["train"] == legacy["train_dataset"]
    assert views["val"] == legacy["val_dataset"]
    assert views["test"] == legacy["test_dataset"]


@needs_refactored_config
@pytest.mark.parametrize("case", list(NEW_OVERRIDES))
def test_relocated_keys_keep_their_values(case):
    legacy = _fixture(case)

    new = _compose(NEW_OVERRIDES[case])

    for legacy_path, new_path in RELOCATED.items():
        if _has(legacy, legacy_path):
            assert _lookup(new, new_path) == _lookup(legacy, legacy_path), legacy_path
    for legacy_path, new_path in RELOCATED_TERMS.items():
        if _has(legacy, legacy_path):
            assert _lookup(new, new_path) == float(_lookup(legacy, legacy_path)), legacy_path
    assert new["objective"]["_target_"] == OBJECTIVE_TARGETS[legacy["model"]["step_loss"]]
    assert new["objective"]["pointwise"]["_target_"] == POINTWISE_TARGETS[legacy["model"]["loss_fn"]]
    assert new["config_version"] == 2


@needs_refactored_config
@pytest.mark.parametrize("case", list(NEW_OVERRIDES))
def test_architecture_keys_stay_in_model(case):
    legacy_model = _fixture(case)["model"]
    expected = {key: value for key, value in legacy_model.items() if key not in MOVED_OUT_OF_MODEL}

    new_model = _compose(NEW_OVERRIDES[case])["model"]
    architecture = {key: value for key, value in new_model.items() if key != "analysis_head"}

    assert architecture == expected


@needs_refactored_config
@pytest.mark.parametrize("case", list(NEW_OVERRIDES))
def test_migrated_fixture_equals_the_new_composition(case):
    migrated = config_utils.migrate_config(_fixture(case))

    new = _compose(NEW_OVERRIDES[case])

    for section in MIGRATED_SECTIONS:
        assert migrated[section] == new[section], section


@needs_refactored_config
@pytest.mark.parametrize("case", list(NEW_OVERRIDES))
def test_migration_is_idempotent(case):
    once = config_utils.migrate_config(_fixture(case))

    twice = config_utils.migrate_config(once)

    assert twice == once
