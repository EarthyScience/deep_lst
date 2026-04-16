"""Parity test: Hydra-resolved config must match the manually merged config.

This test is the definition of done for the Hydra migration (Priority 4).
It will SKIP until conf/ is set up. On Thursday:
  1. Build conf/ with config groups (model/, spatial/, training/, ...)
  2. Remove the skip condition (the `if not CONF_DIR.exists()` guard)
  3. This test must pass before merging the migration branch.

Covers one representative case: mlp_rollout + L_KG_v1 spatial override.
Add more cases here as config groups are added.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = ROOT / "conf"

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from deep_lst.utils import ExperimentPaths  # noqa: E402
from deep_lst.utils.config import merge_configs  # noqa: E402


@pytest.fixture(autouse=True)
def reset_hydra():
    """Hydra keeps global state; clear it between tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _old_config(config_name: str, overrides: list[str]) -> dict:
    """Load config the old way via merge_configs."""
    paths = ExperimentPaths()
    override_paths = [paths.config_root / config_name / "overrides" / f"{o}.yml" for o in overrides]
    return merge_configs(paths.config_path(config_name), override_paths)


def _new_config(overrides: list[str]) -> dict:
    """Load config via Hydra compose and return as plain dict."""
    with initialize_config_dir(config_dir=str(CONF_DIR), job_name="test"):
        cfg = compose(config_name="train", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


# ── parity cases ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "config_name,old_overrides,hydra_overrides",
    [
        (
            "mlp_rollout",
            ["mlp_rollout_L_KG_v1"],
            ["model=mlp_rollout", "stratification=KG", "size=L", "horizon=h3"],
        ),
        (
            "mlp_diag",
            ["mlp_diag_L_KG_v1"],
            ["model=mlp_diag", "stratification=KG", "size=L", "horizon=h3"],
        ),
    ],
)
def test_top_level_keys_match(config_name, old_overrides, hydra_overrides):
    """Every top-level key present in the old config must exist in the Hydra config."""
    old = _old_config(config_name, old_overrides)
    new = _new_config(hydra_overrides)

    # New config has additional group-helper keys (_model_name, _step_loss_*, _loss, etc.)
    # We only require that old keys are present in the new config.
    missing = set(old.keys()) - set(new.keys())
    assert not missing, f"Keys in old config missing from Hydra config: {missing}"


@pytest.mark.parametrize(
    "config_name,old_overrides,hydra_overrides",
    [
        (
            "mlp_rollout",
            ["mlp_rollout_L_KG_v1"],
            ["model=mlp_rollout", "stratification=KG", "size=L", "horizon=h3"],
        ),
        (
            "mlp_diag",
            ["mlp_diag_L_KG_v1"],
            ["model=mlp_diag", "stratification=KG", "size=L", "horizon=h3"],
        ),
    ],
)
def test_model_config_matches(config_name, old_overrides, hydra_overrides):
    old = _old_config(config_name, old_overrides)
    new = _new_config(hydra_overrides)
    assert old["model"] == new["model"], "model config mismatch"


@pytest.mark.parametrize(
    "config_name,old_overrides,hydra_overrides",
    [
        (
            "mlp_rollout",
            ["mlp_rollout_L_KG_v1"],
            ["model=mlp_rollout", "stratification=KG", "size=L", "horizon=h3"],
        ),
    ],
)
def test_training_config_matches(config_name, old_overrides, hydra_overrides):
    old = _old_config(config_name, old_overrides)
    new = _new_config(hydra_overrides)
    assert old["training"] == new["training"], "training config mismatch"


@pytest.mark.parametrize(
    "config_name,old_overrides,hydra_overrides",
    [
        (
            "mlp_rollout",
            ["mlp_rollout_L_KG_v1"],
            ["model=mlp_rollout", "stratification=KG", "size=L", "horizon=h3"],
        ),
    ],
)
def test_train_dataset_config_matches(config_name, old_overrides, hydra_overrides):
    old = _old_config(config_name, old_overrides)
    new = _new_config(hydra_overrides)
    assert old["train_dataset"] == new["train_dataset"], "train_dataset config mismatch"
