"""Rechunk a finished forecast store for fast evaluation reads.

Run this once after all distributed inference array jobs complete. Inference
streams one init_time per chunk (fast, conflict-free writes); this rewrites the
store with init_time chunked into large blocks so evaluation reductions read a
handful of big chunks instead of thousands of tiny ones.

Usage
-----
uv run scripts/inference/run_inference_finalize.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=train_KG_L_fullyear
"""

import hydra
from omegaconf import DictConfig

from deep_lst.evaluation.inference import Forecast
from deep_lst.utils import ExperimentPaths


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference_finalize(cfg: DictConfig) -> None:
    paths = ExperimentPaths()
    output_path = paths.forecast_path(cfg.experiment, cfg.version, cfg.eval_name)

    print(f"Experiment : {cfg.experiment} / {cfg.version}")
    print(f"Finalize   : rechunking {output_path}")

    Forecast.rechunk_store(output_path)


if __name__ == "__main__":
    run_inference_finalize()
