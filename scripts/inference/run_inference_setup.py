"""Create the zarr store for distributed inference without loading model weights.

Run this once before launching the GPU array job (run_inference_array.sh).
The store is sized for the full time range; each array job then fills its chunk.

Usage
-----
uv run scripts/inference/run_inference_setup.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=train_KG_L_fullyear \\
    inference_horizon=200
"""

import hydra
from omegaconf import DictConfig

from deep_lst.evaluation.inference import Forecast
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config
from deep_lst.utils.inference import load_run_config, load_spatial_indices, load_time_indices


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference_setup(cfg: DictConfig) -> None:
    paths = ExperimentPaths()

    config = load_run_config(cfg, paths)
    td_keys = get_keys_from_config(config["test_dataset"])
    spatial_indices = load_spatial_indices(paths, cfg.spatial_indices)
    time_idx = load_time_indices(paths, config["test_dataset"])

    output_path = paths.forecast_path(cfg.experiment, cfg.version, cfg.eval_name)
    seviri_store_path = paths.sampled_store_path("seviri", config["test_dataset"]["spatial"])

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
        seviri_store_path,
    )


if __name__ == "__main__":
    run_inference_setup()
