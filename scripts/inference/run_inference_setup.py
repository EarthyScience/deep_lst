"""Create the zarr store for distributed inference without loading model weights.

Run this once before launching the GPU array job (run_inference_array.sh).
The store is sized for the full time range; each array job then fills its chunk.

Usage
-----
uv run scripts/inference/run_inference_setup.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=train_KG_L_fullyear \\
    inference_horizon=200
"""

import copy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from deep_lst.evaluation.inference import Forecast
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference_setup(cfg: DictConfig) -> None:
    paths = ExperimentPaths()

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

    td_keys = get_keys_from_config(config["test_dataset"])

    spatial_path = paths.spatial_indices_path("seviri", cfg.spatial_indices)
    if not spatial_path.exists():
        raise FileNotFoundError(f"Spatial indices not found: {spatial_path}")
    spatial_indices = np.load(spatial_path)["src_indices"]

    temporal = config["test_dataset"].get("temporal")
    if temporal is None:
        raise ValueError("No temporal period specified in eval config.")
    periods = [temporal] if isinstance(temporal, str) else list(temporal)
    time_idx = np.concatenate(
        [np.load(paths.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in periods]
    )
    time_idx = np.sort(np.unique(time_idx))

    output_path = paths.forecast_path(cfg.experiment, cfg.version, cfg.eval_name)

    print(f"Experiment : {cfg.experiment} / {cfg.version}")
    print(f"Eval       : {cfg.eval_name}  (indices: {cfg.spatial_indices})")
    print(f"Setup      : {len(time_idx)} init times, horizon={cfg.inference_horizon}h → {output_path}")

    Forecast.setup_output(
        output_path,
        td_keys,
        spatial_indices,
        time_idx,
        cfg.inference_horizon,
        config["model"]["step_loss"],
    )


if __name__ == "__main__":
    run_inference_setup()
