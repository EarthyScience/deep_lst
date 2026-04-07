import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as lgrs
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, merge_configs, open_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Name of the experiment config folder.")
parser.add_argument(
    "--tag",
    default=None,
    type=str,
    help="Short label appended to the run name (e.g. --tag lr5e5). Useful for within-family sweeps.",
)
parser.add_argument(
    "--smoke",
    action="store_true",
    help="Smoke-test mode: 1 epoch, 3 train/val/test batches, CSV logger only.",
)
parser.add_argument(
    "--override",
    nargs="+",
    default=None,
    metavar="OVERRIDE",
    help="One or more override file names (without .yml) applied in order. "
    "Files are looked up under configs/<config>/overrides/<name>.yml. "
    "Example: --override mlp_diag mlp_diag_L_KG_v1",
)
args = parser.parse_args()

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


def main():
    paths = ExperimentPaths()
    if args.override:
        override_paths = [paths.config_root / args.config / "overrides" / f"{o}.yml" for o in args.override]
        config = merge_configs(paths.config_path(args.config), override_paths)
    else:
        config = open_config(paths.config_path(args.config))

    # Derive run name: base config + _experiment key set by the spatial override
    if args.override:
        name = f"{args.config}_{config['_experiment']}"
    else:
        name = args.config
    config["name"] = name

    # --- name: apply tag first, then smoke suffix so order is always name_tag_smoke ---
    if args.tag:
        config["name"] = config["name"] + f"_{args.tag}"

    if args.smoke:
        print(">>> SMOKE MODE: overriding training limits.")
        config["training"]["max_epochs"] = 1
        config["training"]["limit_train_batches"] = 3
        config["training"]["limit_val_batches"] = 3
        config["training"]["limit_test_batches"] = 3
        config["training"]["overfit_batches"] = 0.0
        config["wandb_logger"] = False
        config["name"] = config["name"] + "_smoke"

    loggers = []
    if config["csv_logger"]:
        csv_logger = lgrs.CSVLogger(save_dir="./experiments", name=config["name"])
        loggers.append(csv_logger)
    if config["wandb_logger"]:
        wandb_logger = lgrs.WandbLogger(project="deep_lst", name=config["name"])
        loggers.append(wandb_logger)
    if config["profiler"]:
        AdvancedProfiler(dirpath=f"./experiments/{config['name']}", filename="perf_logs")

    # --- snapshot config + metadata into the experiment run directory ---
    if config["csv_logger"]:
        run_dir = Path(csv_logger.log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        from omegaconf import OmegaConf as _OC

        _OC.save(_OC.create(config), run_dir / "config.yml")
        metadata = {
            "config": args.config,
            "tag": args.tag,
            "smoke": args.smoke,
            "overrides": args.override or [],
            "git_hash": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURM_NODELIST"),
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    torch.set_float32_matmul_precision("medium")

    # --- checkpointing: best (by val metric) + last ---
    callbacks = []
    if not args.smoke and config["csv_logger"]:
        monitor = config["model"].get("checkpoint_monitor", "val_loss")
        ckpt_dir = run_dir / "checkpoints"
        callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best",
                monitor=monitor,
                mode="min",
                save_top_k=1,
                save_last=True,
                verbose=True,
            ),
        ]

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
        callbacks=callbacks if callbacks else None,
        accelerator=accelarator,
        devices=devices,
        num_nodes=1,
        strategy="ddp",
        overfit_batches=config["training"]["overfit_batches"],
        limit_train_batches=config["training"]["limit_train_batches"],
        limit_val_batches=config["training"]["limit_val_batches"],
        limit_test_batches=config["training"]["limit_test_batches"],
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=5,
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
