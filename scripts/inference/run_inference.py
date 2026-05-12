"""Run model inference for a trained experiment (or just create the zarr store).

When setup_only=true, creates the output zarr store without loading model weights;
intended for CPU-only submission before the GPU array job.
When setup_only=false (default), runs GPU inference for this job's time chunk.

Usage
-----
# Create zarr store only (CPU, no model weights)
uv run scripts/inference/run_inference.py setup_only=true \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=gobabeb \\
    inference_horizon=200

# Run inference (GPU, reads store created above)
uv run scripts/inference/run_inference.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=gobabeb \\
    inference_horizon=200

# One-command submission via the launcher (recommended):
#   scripts/inference/run_inference.sh --array=0-3 experiment=... version=... eval=... inference_horizon=200
"""

import os

import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.evaluation.inference import Forecast
from deep_lst.mlp.model import load_model
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config
from deep_lst.utils.inference import load_run_config, load_spatial_indices, load_time_indices


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference(cfg: DictConfig) -> None:
    paths = ExperimentPaths()

    config = load_run_config(cfg, paths)
    td_keys = get_keys_from_config(config["test_dataset"])
    spatial_indices = load_spatial_indices(paths, cfg.spatial_indices)
    time_idx = load_time_indices(paths, config["test_dataset"])

    eval_name = cfg.eval_name
    print(f"Experiment : {cfg.experiment} / {cfg.version}")
    print(f"Eval       : {eval_name}  (indices: {cfg.spatial_indices})")

    # --- Load model ---
    lit_module, device = load_model(
        config, td_keys, experiment=cfg.experiment, version=cfg.version, prefer_best=cfg.best
    )

    # --- Distributed inference: compute chunk bounds before building dataset ---
    # n_chunks / chunk_offset can come from Hydra config or from SLURM array env vars.
    # --array=0-N sets SLURM_ARRAY_TASK_COUNT=N+1 and SLURM_ARRAY_TASK_ID per task,
    # so the only submission knob needed is --array=0-N (no n_chunks arg required).

    n_chunks = (
        cfg.n_chunks
        if cfg.n_chunks is not None
        else (int(os.environ["SLURM_ARRAY_TASK_COUNT"]) if "SLURM_ARRAY_TASK_COUNT" in os.environ else None)
    )
    chunk_offset = (
        cfg.chunk_offset
        if cfg.chunk_offset is not None
        else (int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    )
    time_offset = None
    chunk_start = None
    chunk_size = None
    if n_chunks is not None:
        if chunk_offset is None:
            raise ValueError("n_chunks requires chunk_offset (or SLURM_ARRAY_TASK_ID) to be set.")
        chunk_size = int(np.ceil(len(time_idx) / n_chunks))
        chunk_start = chunk_offset * chunk_size
        time_idx = time_idx[chunk_start : chunk_start + chunk_size]
        time_offset = chunk_start
        print(f"Chunk      : {chunk_offset}/{n_chunks} ({len(time_idx)} init times, offset={time_offset})")

    # --- Build dataset and loader ---
    test_dataset = MultimodalDataset.from_modalities(
        modality_ids=config["modality_ids"],
        **config["test_dataset"],
    )

    if n_chunks is not None:
        # Subset dataset to the chunk's time steps only — each job runs only its share.
        # Dataset internal time count may be smaller than the raw time_idx length because
        # _load_samples drops the last `horizon` steps. Use len(test_dataset) as truth.
        n_spatial = len(np.unique(spatial_indices))
        n_total_time = len(test_dataset) // n_spatial
        actual_start = min(chunk_start, n_total_time)
        actual_end = min(chunk_start + chunk_size, n_total_time)
        if actual_start >= n_total_time:
            print(f"Chunk {chunk_offset} has no valid samples (offset beyond dataset), skipping.")
            return
        sample_idx = np.concatenate(
            [np.arange(i * n_total_time + actual_start, i * n_total_time + actual_end) for i in range(n_spatial)]
        )
        test_dataset = Subset(test_dataset, sample_idx.tolist())
        print(f"           : {len(test_dataset)} samples ({actual_end - actual_start} steps × {n_spatial} pixels)")

    batch_size = config["analysis"]["batch_size"]
    num_workers = config["analysis"]["num_workers"]

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=Collator(config["model"]["name"]),
        pin_memory=config["analysis"].get("pin_memory", False),
        persistent_workers=num_workers > 0,
    )

    # --- Run inference ---
    forecast = Forecast(
        model=lit_module,
        device=device,
        loader=test_loader,
        td_keys=td_keys,
        config=config["test_dataset"],
        spatial_indices=spatial_indices,
        prediction_mode=config["model"]["step_loss"],
        apply_consistent_mask=not cfg.no_consistent_mask,
    )
    # --- Save ---
    output_path = paths.forecast_path(cfg.experiment, cfg.version, eval_name)
    forecast.predict_and_save(output_path, time_idx, time_offset=time_offset)


if __name__ == "__main__":
    run_inference()
