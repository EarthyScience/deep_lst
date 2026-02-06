import argparse
import copy

from torch.utils.data import DataLoader

from deep_lst.analysis.forecast import Forecast
from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.mlp.model import load_model
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, open_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment name (config key)")
    parser.add_argument("--version", type=str, required=True, help="Run version folder")
    # Optional time range override for parallelization
    parser.add_argument("--start_time", type=str, default=None, help="Override start time (e.g. '2017-12-15 01:00')")
    parser.add_argument("--end_time", type=str, default=None, help="Override end time (e.g. '2017-12-16 01:00')")
    parser.add_argument("--chunk_id", type=str, default=None, help="Chunk identifier for partial output files")
    parser.add_argument(
        "--no_consistent_mask", action="store_true", help="Disable consistent cloud mask (show all available data)"
    )
    return parser.parse_args()


def run_forecast():
    args = parse_args()
    paths = ExperimentPaths()
    config = open_config(paths.config_path(args.config))

    # Override time range if provided (for parallelization)
    if args.start_time or args.end_time:
        config = copy.deepcopy(config)
        if args.start_time:
            config["test_dataset"]["start_time"] = args.start_time
        if args.end_time:
            config["test_dataset"]["end_time"] = args.end_time

    td_keys = get_keys_from_config(config["test_dataset"])

    # Load Model
    lit_module, device = load_model(config, td_keys, experiment=args.config, version=args.version)

    # Get Dataset and DataLoader
    test_dataset = MultimodalDataset.from_modalities(
        modality_ids=config["modality_ids"],
        **config["test_dataset"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["analysis"]["batch_size"],
        num_workers=config["analysis"]["num_workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=Collator(config["model"]["name"]),
        pin_memory=config["analysis"]["pin_memory"],
        persistent_workers=True,
    )

    forecast = Forecast(
        model=lit_module,
        device=device,
        loader=test_loader,
        td_keys=td_keys,
        config=config["test_dataset"],
        prediction_mode=config["model"]["step_loss"],
        apply_consistent_mask=not args.no_consistent_mask,
    )

    result = forecast.predict()

    # Save with chunk suffix if parallelizing, otherwise normal path
    if args.chunk_id:
        output_path = paths.predictions_path(args.config, args.version, suffix=f"_chunk{args.chunk_id}")
    else:
        output_path = paths.predictions_path(args.config, args.version)

    result.save(output_path)


if __name__ == "__main__":
    run_forecast()
