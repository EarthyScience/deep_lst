"""Run model inference for a trained experiment on an evaluation site.

The saved config.yml from the training run is the authoritative source for model
architecture, features, and normalisation. The Hydra eval group adds the
inference-specific settings (spatial indices, temporal period, batch size) on top.

Usage
-----
# Single eval site
uv run scripts/inference/run_inference.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=gobabeb

# All eval sites via Hydra multirun
uv run scripts/inference/run_inference.py -m \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 \\
    eval=antananarivo,dahra,east_africa,gobabeb,johannesburg,sahel_niger,w_africa

# Training-site inference
uv run scripts/inference/run_inference.py \\
    experiment=mlp_rollout_S_KG_mse_h3_v1 version=version_0 eval=train_KG_S

# Extend rollout horizon beyond training horizon
uv run scripts/inference/run_inference.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=gobabeb \\
    inference_horizon=12

Available eval groups: see conf/eval/
"""

import copy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.evaluation.inference import Forecast
from deep_lst.mlp.model import load_model
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference(cfg: DictConfig) -> None:
    paths = ExperimentPaths()

    # --- Load saved training config (authoritative for model/data) ---
    saved_config_path = paths.experiment_dir(cfg.experiment) / cfg.version / "config.yml"
    if not saved_config_path.exists():
        raise FileNotFoundError(
            f"Saved config not found at {saved_config_path}. "
            "Ensure the experiment version directory contains config.yml."
        )
    saved = OmegaConf.load(saved_config_path)

    # Merge: saved config is base, Hydra cfg provides eval overrides on top.
    # eval group keys (test_dataset.*, analysis.*) win over saved values.
    # Inference-only keys (experiment, version, best, ...) are added as new keys.
    config = OmegaConf.to_container(OmegaConf.merge(saved, cfg), resolve=True)

    eval_name = cfg.eval_name
    print(f"Experiment : {cfg.experiment} / {cfg.version}")
    print(f"Eval       : {eval_name}  (indices: {cfg.spatial_indices})")

    # --- Apply inference_horizon override ---
    trained_horizon = config["test_dataset"]["era5"]["horizon"]
    if cfg.inference_horizon < trained_horizon:
        raise ValueError(f"inference_horizon ({cfg.inference_horizon}) must be >= trained horizon ({trained_horizon}).")
    config = copy.deepcopy(config)
    for mod in config["modality_ids"]:
        config["test_dataset"][mod]["horizon"] = cfg.inference_horizon

    td_keys = get_keys_from_config(config["test_dataset"])

    # --- Load spatial indices ---
    # spatial_indices key in eval config covers both eval regions (eval_gobabeb)
    # and training sites (KG_S) via the model-agnostic spatial_indices_path.
    spatial_path = paths.spatial_indices_path("seviri", cfg.spatial_indices)
    if not spatial_path.exists():
        raise FileNotFoundError(f"Spatial indices not found: {spatial_path}")
    spatial_indices = np.load(spatial_path)["src_indices"]

    # --- Load model ---
    lit_module, device = load_model(
        config, td_keys, experiment=cfg.experiment, version=cfg.version, prefer_best=cfg.best
    )

    # --- Build dataset and loader ---
    test_dataset = MultimodalDataset.from_modalities(
        modality_ids=config["modality_ids"],
        **config["test_dataset"],
    )
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

    # --- Load temporal indices ---
    temporal = config["test_dataset"].get("temporal")
    if temporal is None:
        raise ValueError("No temporal period specified in eval config.")
    periods = [temporal] if isinstance(temporal, str) else list(temporal)
    time_idx = np.concatenate(
        [np.load(paths.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in periods]
    )
    time_idx = np.sort(np.unique(time_idx))

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
    suffix = f"_chunk{cfg.chunk_id}" if cfg.chunk_id else ""
    output_path = paths.forecast_path(cfg.experiment, cfg.version, eval_name, suffix=suffix)
    forecast.predict_and_save(output_path, time_idx)


if __name__ == "__main__":
    run_inference()
