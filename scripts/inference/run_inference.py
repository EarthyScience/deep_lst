"""Run model inference for a trained experiment.

Builds the test dataset, creates the forecast.zarr sized to the dataset's actual
time axis, fills it with predictions in one process, and rechunks at the end.

Usage
-----
uv run scripts/inference/run_inference.py \\
    experiment=mlp_rollout_L_KG_mse_h3_v1 version=version_0 eval=gobabeb \\
    inference_horizon=200
"""

import hydra
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.evaluation.inference import Forecast
from deep_lst.mlp.model import load_model
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, get_packing_keys
from deep_lst.utils.inference import load_run_config


@hydra.main(config_path="../../conf", config_name="infer", version_base="1.3")
def run_inference(cfg: DictConfig) -> None:
    paths = ExperimentPaths()
    config = load_run_config(cfg, paths)
    td_keys = get_keys_from_config(config["test_dataset"])

    lit_module, device = load_model(
        config, td_keys, experiment=cfg.experiment, version=cfg.version, prefer_best=cfg.best
    )

    modality_ids = config["modality_ids"]
    shared = MultimodalDataset.preload_shared(modality_ids, **config["test_dataset"])
    dataset = MultimodalDataset.from_modalities(modality_ids, shared, **config["test_dataset"])
    print(f"Experiment : {cfg.experiment} / {cfg.version}")
    print(f"Eval       : {cfg.eval_name}")
    print(f"Dataset    : {len(dataset.spatial_idx)} pixels × {len(dataset.time_idx)} init times")

    # The streaming zarr write assumes time-major iteration in sorted pixel and time order
    # (Forecast scatters by sorted-unique spatial_idx; init_time slots advance sequentially).
    n_spatial = len(dataset.spatial_idx)
    if not np.array_equal(dataset.indices[:n_spatial, 0], dataset.spatial_idx):
        raise RuntimeError(
            "Dataset pixel order differs from sorted spatial_idx — the forecast.zarr scatter "
            "would assign predictions to wrong pixels. Check the sampled zarr's spatial_idx ordering."
        )
    if not np.array_equal(dataset.indices[::n_spatial, 1], dataset.time_idx):
        raise RuntimeError(
            "Dataset init_time order differs from sorted time_idx — the forecast.zarr init_time "
            "slots would be mislabeled. Check the temporal indices npz ordering."
        )

    loader = DataLoader(
        dataset,
        batch_size=config["analysis"]["batch_size"],
        num_workers=config["analysis"]["num_workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=Collator(config["model"]["name"], get_packing_keys(config["test_dataset"])),
        pin_memory=config["analysis"].get("pin_memory", False),
        persistent_workers=config["analysis"]["num_workers"] > 0,
        prefetch_factor=config["analysis"].get("prefetch_factor", 2),
    )

    forecast = Forecast(
        model=lit_module,
        device=device,
        loader=loader,
        td_keys=td_keys,
        config=config["test_dataset"],
        spatial_indices=dataset.spatial_idx,
        pixel_lonlat=dataset.pixel_lonlat(),
        non_land=dataset.non_land_mask(),
        prediction_mode=config["model"]["step_loss"],
        apply_lw_mask=cfg.land_water_mask,
    )

    output_path = paths.forecast_path(cfg.experiment, cfg.version, cfg.eval_name)
    forecast.predict_and_save(output_path, dataset.time_idx)
    Forecast.rechunk_store(output_path)


if __name__ == "__main__":
    run_inference()
