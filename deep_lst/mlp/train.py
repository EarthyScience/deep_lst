import argparse
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as lgrs
from pytorch_lightning.profilers import AdvancedProfiler
from tensordict.nn import TensorDictModule

from deep_lst.mlp.model import MultiModalMLP
from deep_lst.modules.lightning_modules import LitDataModule, LitModelModule
from deep_lst.utils.misc import open_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Name of the experiment.")
args = parser.parse_args()

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))


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
    config = open_config(experiment=args.config, path="configs")

    loggers = []
    if config["csv_logger"]:
        csv_logger = lgrs.CSVLogger(save_dir=f"./experiments/{config['name']}/logs/", name=config["name"])
        loggers.append(csv_logger)
    if config["wandb_logger"]:
        wandb_logger = lgrs.WandbLogger(project="deep_lst", name=config["name"])
        loggers.append(wandb_logger)
    if config["profiler"]:
        AdvancedProfiler(dirpath=f"./experiments/{config['name']}", filename="perf_logs")

    torch.set_float32_matmul_precision("medium")

    data_module = LitDataModule(config=config)
    data_module.setup()

    in_size, out_size = get_sizes(config["datareader"])

    td_model = TensorDictModule(
        module=MultiModalMLP(in_size, out_size, config["model"]),
        in_keys=[("inputs", "era5"), ("inputs", "seviri")],  # , ("prognostics", "seviri")],
        out_keys=[("predictions", "seviri")],
    )
    model_module = LitModelModule(td_model, config["model"])

    ngpu = torch.cuda.device_count()
    if ngpu != 0:
        print("GPU:", torch.cuda.get_device_name(0))
    print("number of GPUs:", ngpu)

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=loggers,
        accelerator="gpu",
        devices=ngpu,
        num_nodes=1,
        strategy="ddp",
        # overfit_batches=5,  # Use only 5 batches for both train/val
        limit_train_batches=config["training"]["limit_train_batches"],
        limit_val_batches=config["training"]["limit_val_batches"],
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,
    )  # decrease to reduce logging overhead

    print("Trainer world_size:", trainer.world_size, "devices:", torch.cuda.device_count())

    start_time = time.time()
    trainer.fit(model_module, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time) / 60)


if __name__ == "__main__":
    main()
