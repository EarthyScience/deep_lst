import argparse

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
    return parser.parse_args()


def run_forecast():
    args = parse_args()
    paths = ExperimentPaths()
    config = open_config(paths.config_path(args.config))
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
    )

    forecast = Forecast(
        model=lit_module,
        device=device,
        loader=test_loader,
        td_keys=td_keys,
        config=config["test_dataset"],
        prediction_mode=config["model"]["step_loss"],
    )

    result = forecast.predict()

    result.save(paths.predictions_path(args.config, args.version))


if __name__ == "__main__":
    run_forecast()
