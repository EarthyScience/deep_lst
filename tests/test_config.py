"""Tests for the config helpers around the composition (deep_lst/utils/config.py, io.py,
utils/inference.py).

Covered:
  validate_config
    - objective=diagnostic needs a data option without prognostics and with a diagnostic
    - init_fill gated / always needs a diagnostic copy of every prognostic
    - the combinations in use pass
  check_stats_transform
    - statistics computed under another transform than the run applies are refused
  conf/statistics.yaml
    - composes the same `sources` as conf/train.yaml
  conf/infer.yaml and the eval group
    - every eval option overrides splits.test and nothing else
  load_run_config
    - a legacy config.yml is migrated, the eval selection and init_hours land in the test
      view, the statistics tag stays the training one, the horizon is the inference horizon
  LitModelModule
    - builds the objective and the optimiser from the `objective` and `optim` sections
"""

from pathlib import Path

import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.lit_module import LitModelModule
from deep_lst.losses import RolloutLoss
from deep_lst.utils.config import split_views, validate_config
from deep_lst.utils.inference import load_run_config
from deep_lst.utils.io import ExperimentPaths, check_stats_transform

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
FIXTURE_DIR = Path(__file__).parent / "fixtures"
EVAL_OPTIONS = sorted(path.stem for path in (CONF_DIR / "eval").glob("*.yaml"))


def _compose(overrides, config_name="train"):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides)


def _config(overrides, config_name="train"):
    return OmegaConf.to_container(_compose(overrides, config_name), resolve=True)


# ── validate_config ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        ["objective=rollout", "data=full"],
        ["objective=rollout", "data=full_lstdiag"],
        ["objective=rollout", "data=full_joint"],
        ["objective=rollout_gated", "data=full_joint"],
        ["objective=rollout_always", "data=full_joint"],
        ["objective=diagnostic", "data=full_lstdiag"],
    ],
)
def test_the_combinations_in_use_are_valid(overrides):
    validate_config(_config(overrides))


def test_diagnostic_objective_refuses_a_data_option_with_a_prognostic():
    with pytest.raises(ValueError, match="objective=diagnostic needs a data option without prognostics"):
        validate_config(_config(["objective=diagnostic", "data=full"]))


def test_diagnostic_objective_refuses_a_data_option_without_a_diagnostic():
    overrides = ["objective=diagnostic", "data=full_lstdiag", "features.seviri.diagnostics=[]"]
    with pytest.raises(ValueError, match="at least one diagnostic"):
        validate_config(_config(overrides))


@pytest.mark.parametrize("objective", ["rollout_gated", "rollout_always"])
def test_init_fill_refuses_a_data_option_without_a_diagnostic_copy(objective):
    with pytest.raises(ValueError, match="diagnostic copy of every prognostic"):
        validate_config(_config([f"objective={objective}", "data=full"]))


# ── check_stats_transform ─────────────────────────────────────────────────────


def _stats_zarr(tmp_path, transform):
    path = tmp_path / "glofas_v4_stats_test.zarr"
    root = zarr.open_group(str(path), mode="w")
    if transform is not None:
        root.attrs["transform"] = transform
    return path


def test_statistics_with_the_transform_of_the_run_are_accepted(tmp_path):
    check_stats_transform(_stats_zarr(tmp_path, "signed_log1p"), "signed_log1p")


def test_raw_statistics_are_accepted_for_a_source_without_transform(tmp_path):
    check_stats_transform(_stats_zarr(tmp_path, None), None)


def test_raw_statistics_are_refused_for_a_transformed_source(tmp_path):
    with pytest.raises(ValueError, match="transform=None, the run applies transform='signed_log1p'"):
        check_stats_transform(_stats_zarr(tmp_path, None), "signed_log1p")


def test_transformed_statistics_are_refused_for_a_raw_source(tmp_path):
    with pytest.raises(ValueError, match="transform='signed_log1p', the run applies transform=None"):
        check_stats_transform(_stats_zarr(tmp_path, "signed_log1p"), None)


# ── conf/statistics.yaml ──────────────────────────────────────────────────────


def test_statistics_compose_the_sources_of_training():
    statistics = _config(["modality=glofas_v4"], config_name="statistics")

    training = _config([])

    assert statistics["sources"] == training["sources"]
    assert statistics["sources"]["glofas_v4"]["store"] == "glofas_v4_2014_2025"
    assert statistics["sources"]["glofas_v4"]["transform"] == "signed_log1p"
    assert statistics["sources"]["seviri"]["increment_vars"] == ["LST", "MLE", "MH"]
    assert statistics["spatial"] == training["splits"]["train"]["stats_spatial"]


# ── conf/infer.yaml and the eval group ────────────────────────────────────────


@pytest.mark.parametrize("eval_option", EVAL_OPTIONS)
def test_eval_options_override_the_test_split_only(eval_option):
    default = _config(["eval=gobabeb", "experiment=run"], config_name="infer")

    config = _config([f"eval={eval_option}", "experiment=run"], config_name="infer")

    assert sorted(config) == sorted(default)
    assert set(config["splits"]) == {"test"}
    assert set(config["splits"]["test"]) == {"spatial", "temporal", "anchor", "init_hours"}
    assert config["eval_name"] == eval_option


# ── load_run_config ───────────────────────────────────────────────────────────


@pytest.fixture
def legacy_run(tmp_path):
    """An experiments/ root holding one run whose config.yml has the layout before version 2."""
    run_dir = tmp_path / "legacy_run" / "version_0"
    run_dir.mkdir(parents=True)
    fixture = (FIXTURE_DIR / "config_v1_default.yml").read_text()
    (run_dir / "config.yml").write_text(fixture)
    return ExperimentPaths(experiment_root=tmp_path)


def test_load_run_config_migrates_a_legacy_run_and_overlays_the_eval(legacy_run):
    cfg = _compose(["eval=gobabeb", "experiment=legacy_run", "inference_horizon=240"], config_name="infer")

    config = load_run_config(cfg, legacy_run)

    assert config["config_version"] == 2
    assert config["window"] == {"lookback": 168, "horizon": 240}
    assert config["objective"]["kind"] == "rollout"
    assert config["objective"]["init_fill"] == "none"
    test_view = split_views(config)["test"]
    assert test_view["spatial"] == "eval_gobabeb"
    assert test_view["temporal"] == "test"
    assert test_view["anchor"] == "seviri"
    assert test_view["stats_spatial"] == "mtg_new"
    assert test_view["init_hours"] == [0, 6, 12, 18]
    assert test_view["seviri"]["horizon"] == 240
    assert test_view["seviri"]["lookback"] == 168
    assert test_view["seviri"]["prognostics"] == ["LST"]


def test_load_run_config_refuses_a_horizon_below_the_trained_one(legacy_run):
    cfg = _compose(["eval=gobabeb", "experiment=legacy_run", "inference_horizon=12"], config_name="infer")

    with pytest.raises(ValueError, match="must be >= trained horizon"):
        load_run_config(cfg, legacy_run)


# ── LitModelModule ────────────────────────────────────────────────────────────


def test_lit_module_builds_objective_and_optimiser_from_the_config():
    config = _config(["objective=rollout_gated", "loss=mae", "data=full_joint"])
    keys = {"prognostic_keys": [("seviri", "LST")], "diagnostic_keys": [("seviri", "LST")]}

    module = LitModelModule(torch.nn.Linear(1, 1), config, **keys)

    objective = module.step_loss
    assert isinstance(objective, RolloutLoss)
    assert objective.pointwise.func is torch.nn.functional.l1_loss
    assert objective.init_fill == "gated"
    assert objective.detach_rollout is False
    assert objective.weight_teacher_forcing == 1.0
    assert objective.weight_rollout == 1.0
    assert objective.weight_increment == 0.0
    assert objective.checkpoint_monitor == "val_loss_step_epoch_log"
    optimizer = module.configure_optimizers()["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["weight_decay"] == 0.0001
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.param_groups[0]["initial_lr"] == 3.7e-4
