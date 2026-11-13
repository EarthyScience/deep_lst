"""Golden tests for the Hydra configuration layout (config_version 2).

tests/fixtures/config_v1_<case>.yml hold what five legacy command lines composed at develop
5274122 (`train.py ... --cfg job --resolve`), before the groups objective / optim / sources
and the sections window / features / splits existed. They are the reference for "the
refactor changed where a value lives, not the value" and are never regenerated.

For every case the new command line has to give
  - the same three per-split dataset dicts (split_views), in the same modality order
  - the same value for every key that moved (RELOCATED, RELOCATED_TERMS)
  - the same architecture under `model`
  - the same value for every key that did not move
and migrate_config has to lift the legacy fixture to what the new command line composes.

Where experiments/ exists (not in CI), every saved config.yml is migrated as well.
"""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.lit_module import build_objective
from deep_lst.utils import config as config_utils

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
FIXTURE_DIR = Path(__file__).parent / "fixtures"

# case -> the command line that composes today what the fixture's legacy command line composed
# (the legacy command line is in the header of each fixture)
NEW_OVERRIDES = {
    "default": ["model=mlp", "objective=rollout", "loss=smoothl1", "horizon=h24", "data=full"],
    "transformer": ["model=transformer", "objective=rollout", "loss=smoothl1", "horizon=h24", "data=full"],
    "mlp_gated": ["model=mlp", "objective=rollout_gated", "loss=smoothl1", "horizon=h24", "data=full_joint"],
    "h72_noera5": ["model=mlp", "objective=rollout", "loss=smoothl1", "horizon=h72", "data=noera5"],
    "lstdiag": ["model=mlp", "objective=rollout", "loss=smoothl1", "horizon=h24", "data=full_lstdiag"],
}
NEW_RUN_NAMES = {
    "default": "v18_mlp_rollout_smoothl1_h24_full",
    "transformer": "v18_transformer_rollout_smoothl1_h24_full",
    "mlp_gated": "v18_mlp_rollout_gated_smoothl1_h24_full_joint",
    "h72_noera5": "v18_mlp_rollout_smoothl1_h72_noera5",
    "lstdiag": "v18_mlp_rollout_smoothl1_h24_full_lstdiag",
}

# legacy path -> new path; the value must be equal
RELOCATED = {
    "model.learning_rate": "optim.learning_rate",
    "model.weight_decay": "optim.weight_decay",
    "model.warmup_steps": "optim.warmup_steps",
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
    "model.rollout_loss": "objective.terms.rollout",
}
# legacy model.loss_fn -> objective.pointwise._target_
POINTWISE_TARGETS = {
    "smooth_l1": "torch.nn.functional.smooth_l1_loss",
    "mse": "torch.nn.functional.mse_loss",
    "mae": "torch.nn.functional.l1_loss",
}
# keys that left `model`: moved (above) or deleted because nothing read them (the last two)
MOVED_OUT_OF_MODEL = [
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "detach_rollout",
    "init_fill",
    "init_hide_fraction",
    "checkpoint_monitor",
    "teacher_forcing_loss",
    "incremental_loss",
    "rollout_loss",
    "step_loss",
    "loss_fn",
    "scheduler_T0",
    "probabilistic",
]
# legacy top-level keys without a counterpart: the dataset dicts (now split_views), the helper
# keys they interpolated, the run-name tokens (now experiment_name) and two never-read keys
DROPPED_TOP_LEVEL = [
    "train_dataset",
    "val_dataset",
    "test_dataset",
    "_lookback",
    "_seviri_store",
    "_era5_store",
    "_era5land_store",
    "_glofas_store",
    "_seviri_load",
    "_era5_load",
    "_era5land_load",
    "_glofas_load",
    "_glofas_transform",
    "_experiment",
    "_model_name",
    "_loss",
    "_horizon",
    "_data",
    "_sigma_prognostics",
    "_sigma_diagnostics",
]
NEW_TOP_LEVEL = ["config_version", "paths", "objective", "optim", "window", "sources", "features", "splits"]
# the sections migrate_config has to rebuild exactly
MIGRATED_SECTIONS = [
    "config_version",
    "modality_ids",
    "model",
    "objective",
    "optim",
    "window",
    "features",
    "splits",
    "training",
]

CASES = list(NEW_OVERRIDES)


def _fixture(case):
    return yaml.safe_load((FIXTURE_DIR / f"config_v1_{case}.yml").read_text())


def _compose(overrides, config_name="train"):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def _lookup(config, dotted_path):
    node = config
    for key in dotted_path.split("."):
        node = node[key]
    return node


def _modalities(view):
    return [key for key, value in view.items() if isinstance(value, dict)]


@pytest.mark.parametrize("case", CASES)
def test_split_views_reproduce_the_legacy_datasets(case):
    legacy = _fixture(case)

    views = config_utils.split_views(_compose(NEW_OVERRIDES[case]))

    assert views["train"] == legacy["train_dataset"]
    assert views["val"] == legacy["val_dataset"]
    assert views["test"] == legacy["test_dataset"]
    # dict equality ignores order; the modality order fixes the channel order of the tensors
    assert _modalities(views["train"]) == _modalities(legacy["train_dataset"])
    new_keys = config_utils.get_packing_keys(views["train"])
    legacy_keys = config_utils.get_packing_keys(legacy["train_dataset"])
    assert new_keys == legacy_keys


@pytest.mark.parametrize("case", CASES)
def test_relocated_keys_keep_their_values(case):
    legacy = _fixture(case)

    new = _compose(NEW_OVERRIDES[case])

    for legacy_path, new_path in RELOCATED.items():
        assert _lookup(new, new_path) == _lookup(legacy, legacy_path), legacy_path
    for legacy_path, new_path in RELOCATED_TERMS.items():
        assert _lookup(new, new_path) == float(_lookup(legacy, legacy_path)), legacy_path
    assert legacy["model"]["step_loss"] == "RolloutLoss"
    assert new["objective"]["_target_"] == "deep_lst.losses.RolloutLoss"
    assert new["objective"]["pointwise"]["_target_"] == POINTWISE_TARGETS[legacy["model"]["loss_fn"]]
    assert new["config_version"] == 2


@pytest.mark.parametrize("case", CASES)
def test_architecture_keys_stay_in_model(case):
    legacy_model = _fixture(case)["model"]
    expected = {key: value for key, value in legacy_model.items() if key not in MOVED_OUT_OF_MODEL}

    new_model = _compose(NEW_OVERRIDES[case])["model"]

    assert new_model == expected


@pytest.mark.parametrize("case", CASES)
def test_every_other_key_is_unchanged(case):
    legacy = _fixture(case)
    untouched = [key for key in legacy if key not in DROPPED_TOP_LEVEL and key != "model"]

    new = _compose(NEW_OVERRIDES[case])

    for key in untouched:
        assert new[key] == legacy[key], key
    assert sorted(new) == sorted(untouched + ["model"] + NEW_TOP_LEVEL)


def test_training_path_overrides_leave_scientific_config_unchanged():
    default = _compose(NEW_OVERRIDES["default"])
    overridden = _compose(
        [*NEW_OVERRIDES["default"], "paths.eozarr_root=local-data", "paths.preload_cache_root=local-cache"]
    )
    assert default.pop("paths") == {"eozarr_root": None, "preload_cache_root": None}
    assert overridden.pop("paths") == {"eozarr_root": "local-data", "preload_cache_root": "local-cache"}
    assert overridden == default


def test_modality_store_root_reaches_each_split_without_changing_other_settings():
    default = _compose(NEW_OVERRIDES["default"])
    overridden = _compose([*NEW_OVERRIDES["default"], "+sources.seviri.store_root=local-seviri"])
    for view in config_utils.split_views(overridden).values():
        assert view["seviri"]["store_root"] == "local-seviri"
        assert "store_root" not in view["era5"]
        assert "store_root" not in view["era5_land"]
    assert overridden["sources"]["seviri"].pop("store_root") == "local-seviri"
    assert overridden == default


@pytest.mark.parametrize("case", CASES)
def test_migrated_fixture_equals_the_new_composition(case):
    migrated = config_utils.migrate_config(_fixture(case))

    new = _compose(NEW_OVERRIDES[case])

    for section in MIGRATED_SECTIONS:
        assert migrated[section] == new[section], section
    # a legacy config knows nothing of increment_vars, which only compute_statistics.py reads
    for modality, source in new["sources"].items():
        expected = {key: value for key, value in source.items() if key != "increment_vars"}
        assert migrated["sources"][modality] == expected, modality
    assert list(migrated["sources"]) == list(new["sources"])


@pytest.mark.parametrize("case", CASES)
def test_migration_is_idempotent(case):
    once = config_utils.migrate_config(_fixture(case))

    twice = config_utils.migrate_config(once)

    assert twice == once


def test_a_current_config_passes_migration_unchanged():
    new = _compose(NEW_OVERRIDES["default"])

    assert config_utils.migrate_config(new) is new


def test_migration_refuses_a_config_that_is_not_a_rollout_run():
    with pytest.raises(ValueError, match="RolloutLoss"):
        config_utils.migrate_config({"model": {"step_loss": "DiagnosticMSE"}})


@pytest.mark.parametrize("case", CASES)
def test_run_name_is_built_from_the_chosen_options(case):
    overrides = NEW_OVERRIDES[case]
    choices = dict(override.split("=") for override in overrides)

    name = config_utils.experiment_name(_compose(overrides), choices)

    assert name == NEW_RUN_NAMES[case]


def test_diagnostic_objective_is_the_rollout_objective_without_the_rollout_branch():
    data = ["data=full_lstdiag"]
    rollout = _compose(["objective=rollout"] + data)

    diagnostic = _compose(["objective=diagnostic"] + data)

    expected = dict(rollout["objective"])
    expected["kind"] = "diagnostic"
    expected["terms"] = {"teacher_forcing": 1.0, "rollout": 0.0, "increment": 0.0}
    expected["checkpoint_monitor"] = "val_loss_step_epoch"
    assert diagnostic["objective"] == expected
    for section in ("model", "optim", "window", "sources", "features", "splits"):
        assert diagnostic[section] == rollout[section], section


SAVED_CONFIGS = sorted((REPO_ROOT / "experiments").glob("*/version_*/config.yml"))


@pytest.mark.skipif(not SAVED_CONFIGS, reason="no experiments/ in this working tree")
def test_every_saved_config_migrates_to_its_own_datasets():
    """Old runs keep working: the migrated config gives back the dataset dicts the run was
    trained with, in the same channel order, and its objective can be built."""
    for path in SAVED_CONFIGS:
        saved = yaml.safe_load(path.read_text())

        migrated = config_utils.migrate_config(saved)

        views = config_utils.split_views(migrated)
        assert views["train"] == saved["train_dataset"], path
        assert views["val"] == saved["val_dataset"], path
        assert views["test"] == saved["test_dataset"], path
        new_keys = config_utils.get_packing_keys(views["train"])
        assert new_keys == config_utils.get_packing_keys(saved["train_dataset"]), path
        td_keys = config_utils.get_keys_from_config(views["train"])
        objective = build_objective(migrated["objective"], **td_keys)
        assert objective.init_fill == saved["model"].get("init_fill", "none"), path
        assert objective.detach_rollout == saved["model"]["detach_rollout"], path
        assert objective.weight_increment == float(saved["model"]["incremental_loss"]), path
