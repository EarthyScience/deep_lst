"""Finetune a pre-trained MLP on a continuation config (e.g. horizon scheduling).

Loads weights from a pre-trained checkpoint and continues training with a new
config — typically a longer horizon, adjusted timestep_weights, and/or lower lr.
Architecture must be identical between pre-training and finetuning runs.

Usage
-----
uv run deep_lst/mlp/finetune.py \\
    --from mlp_rollout_L_KG_v1/version_1 \\
    --config mlp_rollout \\
    --override mlp_rollout_L_KG_v1 mlp_rollout_L_KG_v1_h12

The run name is derived from the config + override's _experiment key (e.g.
mlp_rollout_L_KG_v1_h12), matching the naming convention of train.py.

Override naming convention: {base_experiment}_h{N}.yml sets horizon, timestep_weights,
and optionally lr and max_epochs.
"""

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

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config, merge_configs


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        dest="from_run",
        required=True,
        metavar="EXPERIMENT/VERSION",
        help="Pre-trained run to load weights from. Format: experiment_name/version_N. "
        "Loads best.ckpt (falls back to last.ckpt).",
    )
    parser.add_argument("--config", required=True, type=str, help="Base config folder name (e.g. mlp_rollout).")
    parser.add_argument(
        "--override",
        nargs="+",
        default=None,
        metavar="OVERRIDE",
        help="Override file names applied in order. Must include the spatial override and "
        "the horizon continuation override (e.g. mlp_rollout_L_KG_v1 mlp_rollout_L_KG_v1_h12).",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Which checkpoint to load from the source run (default: best).",
    )
    parser.add_argument("--tag", default=None, type=str, help="Short label appended to the run name.")
    parser.add_argument("--smoke", action="store_true", help="Smoke-test mode: 1 epoch, 3 batches, CSV only.")
    args = parser.parse_args()

    print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
    print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))

    paths = ExperimentPaths()

    # --- resolve source checkpoint ---
    from_parts = args.from_run.split("/")
    if len(from_parts) != 2:
        raise ValueError(f"--from must be EXPERIMENT/VERSION, got: {args.from_run!r}")
    from_experiment, from_version = from_parts

    if args.checkpoint == "best":
        ckpt_path = paths.best_checkpoint(from_experiment, from_version)
    else:
        ckpt_path = paths.checkpoint_dir(from_experiment, from_version) / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"last.ckpt not found at {ckpt_path}")

    print(f"Loading weights from: {ckpt_path}")

    # --- load new config ---
    if args.override:
        override_paths = [paths.config_root / args.config / "overrides" / f"{o}.yml" for o in args.override]
        config = merge_configs(paths.config_path(args.config), override_paths)
    else:
        raise ValueError("--override is required for finetune.py (need at least the spatial + horizon overrides)")

    name = f"{args.config}_{config['_experiment']}"
    config["name"] = name

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

    # --- loggers ---
    loggers = []
    if config["csv_logger"]:
        csv_logger = lgrs.CSVLogger(save_dir="./experiments", name=config["name"])
        loggers.append(csv_logger)
    if config["wandb_logger"]:
        wandb_logger = lgrs.WandbLogger(project="deep_lst", name=config["name"])
        loggers.append(wandb_logger)

    # --- snapshot config + metadata ---
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
            "from_run": args.from_run,
            "from_checkpoint": str(ckpt_path),
            "git_hash": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURM_NODELIST"),
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    torch.set_float32_matmul_precision("medium")

    # --- checkpointing ---
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

    # --- build model from new config, then load pre-trained weights ---
    data_module = LitDataModule(**config)
    data_module.setup()

    td_keys = get_keys_from_config(config["train_dataset"])

    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
    )
    lit_module = LitModelModule(model, config, **td_keys)

    state = torch.load(ckpt_path, map_location="cpu")
    lit_module.load_state_dict(state["state_dict"])
    print(f"Weights loaded. Starting finetuning as: {config['name']}")

    # --- trainer ---
    ngpu = torch.cuda.device_count()
    accelerator = "gpu" if ngpu > 0 else "cpu"
    devices = ngpu if ngpu > 0 else 1

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=loggers,
        callbacks=callbacks if callbacks else None,
        accelerator=accelerator,
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

    start = time.time()
    trainer.fit(lit_module, datamodule=data_module)
    print(f"Finetuning completed in {(time.time() - start) / 60:.1f} minutes.")

    start = time.time()
    test_out = trainer.test(lit_module, datamodule=data_module)
    if config["wandb_logger"]:
        test_loss = float(test_out[0]["test_loss_epoch"]) / float(test_out[0]["test_valid_count"])
        wandb_logger.experiment.summary["test_loss"] = test_loss
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
