import argparse
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as lgrs
from pytorch_lightning.profilers import AdvancedProfiler

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, open_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Name of the experiment.")
args = parser.parse_args()

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))


def main():
    paths = ExperimentPaths()
    config = open_config(paths.config_path(args.config))

    loggers = []
    if config["csv_logger"]:
        csv_logger = lgrs.CSVLogger(save_dir="./experiments", name=config["name"])
        loggers.append(csv_logger)
    if config["wandb_logger"]:
        wandb_logger = lgrs.WandbLogger(project="deep_lst", name=config["name"])
        loggers.append(wandb_logger)
    if config["profiler"]:
        AdvancedProfiler(dirpath=f"./experiments/{config['name']}", filename="perf_logs")

    torch.set_float32_matmul_precision("medium")

    data_module = LitDataModule(**config)
    data_module.setup()

    td_keys = get_keys_from_config(config["train_dataset"])

    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
    )
    model_module = LitModelModule(model, config, **td_keys)

    ngpu = torch.cuda.device_count()
    print("number of GPUs:", ngpu)
    if ngpu != 0:
        print("GPU:", torch.cuda.get_device_name(0))
        accelarator = "gpu"
        devices = ngpu
    else:
        accelarator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=loggers,
        accelerator=accelarator,
        devices=devices,
        num_nodes=1,
        strategy="ddp",
        overfit_batches=config["training"]["overfit_batches"],
        limit_train_batches=config["training"]["limit_train_batches"],
        limit_val_batches=config["training"]["limit_val_batches"],
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,  # decrease to reduce logging overhead
    )

    start_time = time.time()
    trainer.fit(model_module, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time) / 60)

    start_time = time.time()
    test_out = trainer.test(model_module, datamodule=data_module)
    if config["wandb_logger"]:
        test_loss = float(test_out[0]["test_loss_epoch"]) / float(test_out[0]["test_valid_count"])
        wandb_logger.experiment.summary["test_loss"] = test_loss
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
