"""Shared helpers for inference scripts."""

import copy

from omegaconf import DictConfig, OmegaConf

from deep_lst.utils import ExperimentPaths


def load_run_config(cfg: DictConfig, paths: ExperimentPaths) -> dict:
    """Load saved training config, merge with Hydra eval overrides, apply horizon override.

    Returns the fully resolved config dict ready for dataset/model construction.
    Raises FileNotFoundError if the saved config is missing.
    Raises ValueError if inference_horizon is smaller than the trained horizon.
    """
    saved_config_path = paths.experiment_dir(cfg.experiment) / cfg.version / "config.yml"
    if not saved_config_path.exists():
        raise FileNotFoundError(
            f"Saved config not found at {saved_config_path}. "
            "Ensure the experiment version directory contains config.yml."
        )
    saved = OmegaConf.load(saved_config_path)
    config = OmegaConf.to_container(OmegaConf.merge(saved, cfg), resolve=True)

    trained_horizon = config["test_dataset"]["era5"]["horizon"]
    if cfg.inference_horizon < trained_horizon:
        raise ValueError(f"inference_horizon ({cfg.inference_horizon}) must be >= trained horizon ({trained_horizon}).")
    config = copy.deepcopy(config)
    for mod in config["modality_ids"]:
        config["test_dataset"][mod]["horizon"] = cfg.inference_horizon

    return config
