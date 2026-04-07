"""Run model inference for a trained experiment on an evaluation site.

Usage:
    python scripts/inference/run_inference.py \
        --experiment mlp_diag_L_KG_v1 \
        --version version_0 \
        --eval antananarivo

The script reads metadata.json to recover the base config and spatial/temporal
overrides used during training, then adds the eval override on top.
"""

import argparse
import copy
import json

import numpy as np
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.evaluation.inference import Forecast
from deep_lst.mlp.model import load_model
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, merge_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name, e.g. mlp_diag_L_KG_v1")
    parser.add_argument("--version", type=str, required=True, help="Version folder, e.g. version_0")
    parser.add_argument(
        "--eval", type=str, required=True, help="Eval site name (without eval_ prefix), e.g. antananarivo"
    )
    parser.add_argument("--start_time", type=str, default=None, help="Override start time for parallelisation")
    parser.add_argument("--end_time", type=str, default=None, help="Override end time for parallelisation")
    parser.add_argument("--chunk_id", type=str, default=None, help="Chunk identifier for partial output files")
    parser.add_argument("--no_consistent_mask", action="store_true", help="Disable consistent cloud mask")
    parser.add_argument("--best", action="store_true", help="Load best.ckpt instead of last.ckpt")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers from config")
    parser.add_argument(
        "--temporal",
        nargs="+",
        default=None,
        metavar="PERIOD",
        help="One or more temporal period names to run inference over, "
        "e.g. --temporal test_period_00 test_period_03. "
        "Overrides temporal setting in config.",
    )
    return parser.parse_args()


def run_inference():
    args = parse_args()
    paths = ExperimentPaths()

    # --- Recover base config + overrides from training metadata ---
    metadata_file = paths.metadata_path(args.experiment, args.version)
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_file}. Pass --experiment as the exact experiment directory name."
        )
    with open(metadata_file) as f:
        metadata = json.load(f)

    config_name = metadata["config"]  # e.g. "mlp_diag"
    train_overrides = metadata["overrides"]  # e.g. ["L_KG_v1"]
    eval_override = f"eval_{args.eval}"

    all_overrides = train_overrides + [eval_override]
    override_paths = [paths.config_root / config_name / "overrides" / f"{o}.yml" for o in all_overrides]
    config = merge_configs(paths.config_path(config_name), override_paths)

    eval_name = config.get("_eval_name", args.eval)

    # --- Optional time range override for parallelisation ---
    if args.start_time or args.end_time or args.temporal:
        config = copy.deepcopy(config)
        if args.start_time:
            config["test_dataset"]["start_time"] = args.start_time
        if args.end_time:
            config["test_dataset"]["end_time"] = args.end_time
        if args.temporal:
            config["test_dataset"]["temporal"] = args.temporal[0] if len(args.temporal) == 1 else args.temporal

    td_keys = get_keys_from_config(config["test_dataset"])

    # --- Load spatial indices for the eval site ---
    spatial_path = paths.eval_indices_path("seviri", eval_name)
    if not spatial_path.exists():
        raise FileNotFoundError(f"Eval spatial indices not found: {spatial_path}")
    spatial_indices = np.load(spatial_path)["indices"]

    # --- Load model ---
    lit_module, device = load_model(
        config, td_keys, experiment=args.experiment, version=args.version, prefer_best=args.best
    )

    # --- Build dataset and loader ---
    test_dataset = MultimodalDataset.from_modalities(
        modality_ids=config["modality_ids"],
        **config["test_dataset"],
    )
    batch_size = args.batch_size or config["analysis"]["batch_size"]
    num_workers = args.num_workers or config["analysis"]["num_workers"]

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

    # --- Load time indices from temporal period files ---
    temporal = config["test_dataset"].get("temporal")
    if temporal is None:
        raise ValueError("No temporal period specified. Pass --temporal or set 'temporal' in the eval config.")
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
        apply_consistent_mask=not args.no_consistent_mask,
    )
    result = forecast.predict_with_time_index(time_idx)

    # --- Save ---
    suffix = f"_chunk{args.chunk_id}" if args.chunk_id else ""
    output_path = paths.forecast_path(args.experiment, args.version, eval_name, suffix=suffix)
    result.save(output_path)
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    run_inference()
