"""Finetune a pre-trained MLP on a new config (e.g. horizon scheduling).

Loads weights from a pre-trained checkpoint and continues training with a new
Hydra-composed config — typically a longer horizon, adjusted timestep_weights,
and/or lower lr. Architecture must be identical between pre-training and finetuning.

Usage
-----
uv run deep_lst/mlp/finetune.py \\
    model=mlp_rollout stratification=KG size=L loss=crps horizon=h12 \\
    from_run=mlp_rollout_L_KG_mse_h3_v1/version_0

Override any single value on the CLI:
    uv run deep_lst/mlp/finetune.py ... training.max_epochs=40 checkpoint=last
"""

import json
import os
import subprocess
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as lgrs
from pytorch_lightning.callbacks import ModelCheckpoint

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.config import get_keys_from_config
from deep_lst.utils.io import load_inc_variances

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


@hydra.main(config_path="../../conf", config_name="finetune", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)

    smoke = config.pop("smoke", False)
    tag = config.pop("tag", None)
    from_run = config.pop("from_run")
    checkpoint_choice = config.pop("checkpoint", "best")

    name = f"{config['_model_name']}_{config['_experiment']}"
    if tag:
        name = f"{name}_{tag}"

    if smoke:
        print(">>> SMOKE MODE: overriding training limits.")
        config["training"]["max_epochs"] = 1
        config["training"]["limit_train_batches"] = 3
        config["training"]["limit_val_batches"] = 3
        config["training"]["limit_test_batches"] = 3
        config["training"]["overfit_batches"] = 0.0
        config["wandb_logger"] = False
        name = f"{name}_smoke"
    config["name"] = name

    # --- resolve source checkpoint ---
    paths = ExperimentPaths()
    from_parts = from_run.split("/")
    if len(from_parts) != 2:
        raise ValueError(f"from_run must be EXPERIMENT/VERSION, got: {from_run!r}")
    from_experiment, from_version = from_parts

    if checkpoint_choice == "best":
        ckpt_path = paths.best_checkpoint(from_experiment, from_version)
    else:
        ckpt_path = paths.checkpoint_dir(from_experiment, from_version) / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"last.ckpt not found at {ckpt_path}")

    print(f"Loading weights from: {ckpt_path}")

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
        OmegaConf.save(OmegaConf.create(config), run_dir / "config.yml")
        metadata = {
            "name": name,
            "config": config["_model_name"],
            "smoke": smoke,
            "tag": tag,
            "from_run": from_run,
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
    if not smoke and config["csv_logger"]:
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

    inc_variances: dict[str, float] = {}
    for modality, mod_cfg in config["train_dataset"].items():
        if isinstance(mod_cfg, dict) and "statistics" in mod_cfg:
            inc_variances.update(load_inc_variances(paths.statistics_path(modality, mod_cfg["statistics"])))

    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
    )
    lit_module = LitModelModule(
        model,
        config,
        **td_keys,
        inc_variances=inc_variances,
        incremental_loss=config["model"].get("incremental_loss", False),
    )

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
        test_loss = test_out[0].get("test_loss_step", test_out[0].get("test_loss"))
        if test_loss is not None:
            wandb_logger.experiment.summary["test_loss"] = float(test_loss)
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
