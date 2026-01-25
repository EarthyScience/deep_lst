"""
This script evaluates a trained ResNet model on a test dataset and computes relevant metrics.
It loads the model and data configuration, sets up the data module and model,
and runs the evaluation, logging results to CSV and Weights & Biases.
"""

import argparse
import code
import glob
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers
from pytorch_lightning.profilers import AdvancedProfiler
from tensordict.nn import TensorDictModule

from deep_lst.dataset.datamodule import DataModule
from deep_lst.modules.lightning_modules import LitModelModule
from deep_lst.resnet.model import TinyResNet
from deep_lst.utils.misc import open_config

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, type=str, help="Name of the experiment.")
parser.add_argument("--version", required=True, type=str, help="Experiment version to load.")
args = parser.parse_args()


def get_sizes(config):
    in_size = len(
        config["era5"]["inputs"]
        + config["seviri"]["inputs"]
        + config["era5"]["prognostics"]
        + config["seviri"]["prognostics"]
    )
    out_size = len(
        config["era5"]["prognostics"]
        + config["seviri"]["prognostics"]
        + config["era5"]["diagnostics"]
        + config["seviri"]["diagnostics"]
    )
    return in_size, out_size


def main():
    config = open_config(experiment=args.experiment_name, path="experiments")

    csv_logger = loggers.CSVLogger(save_dir=f"./experiments/{args.experiment_name}/logs/", name=config["name"])
    wandb_logger = loggers.WandbLogger(project="deep_lst", name=config["name"])

    profiler = AdvancedProfiler(dirpath=f"./experiments/{args.experiment_name}", filename="perf_logs")

    torch.set_float32_matmul_precision("medium")

    data_module = DataModule(config=config)
    data_module.setup()

    in_size, out_size = get_sizes(config["datareader"])

    td_model = TensorDictModule(
        module=TinyResNet(in_size, out_size, **config["model"]),
        in_keys=[("inputs", "era5"), ("inputs", "seviri")],
        out_keys=[("predictions", "seviri")],
    )

    ckpt_glob = f"./experiments/{args.experiment_name}/logs/{args.experiment_name}/{args.version}/checkpoints/*.ckpt"
    ckpt_path = sorted(glob.glob(ckpt_glob))[-1]  # or pick by mtime
    print("Loading:", ckpt_path)

    trained_model = LitModelModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", model=td_model, config=config["model"]
    )

    trainer = pl.Trainer(
        logger=[csv_logger, wandb_logger],
        accelerator="auto",
        devices="auto",
        max_epochs=config["training"]["epochs"],
        profiler=profiler,
    )

    print("Starting evaluation...")
    start_time = time.time()
    test_out = trainer.test(trained_model, datamodule=data_module)
    end_time = time.time()
    code.interact(local=locals())

    test_loss = float(test_out[0]["test_loss_epoch"]) / float(test_out[0]["test_valid_count"])
    wandb_logger.experiment.summary["test_loss"] = test_loss

    print(f"Evaluation completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
