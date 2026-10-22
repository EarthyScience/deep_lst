"""
To test with resolved config: python deep_lst/train.py model=mlp_rollout loss=smoothl1 horizon=h24 data=full --cfg job --resolve
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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler  # noqa: F401

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.lit_module import LitModelModule
from deep_lst.utils.config import build_model, get_keys_from_config, get_packing_keys
from deep_lst.utils.io import ExperimentPaths, load_inc_variances, stats_spatial

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


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    paths = ExperimentPaths()

    config = OmegaConf.to_container(cfg, resolve=True)
    pl.seed_everything(config["seed"], workers=True)

    smoke = config.pop("smoke", False)
    tag = config.pop("tag", None)
    hpo_path = config.pop("hpo_path", None)
    use_hpo = config.pop("hpo", False)

    if use_hpo and hpo_path:
        _HPO_PARAM_MAP = {
            "learning_rate": ("model", "learning_rate"),
            "dropout": ("model", "dropout"),
            "weight_decay": ("model", "weight_decay"),
            "batch_size": ("training", "batch_size"),
        }
        with open(hpo_path) as f:
            hpo = json.load(f)
        params = hpo.get("params", {})
        for hpo_key, (section, cfg_key) in _HPO_PARAM_MAP.items():
            if hpo_key in params:
                config[section][cfg_key] = params[hpo_key]
        if "hidden_sizes" in hpo:
            config["model"]["hidden_sizes"] = hpo["hidden_sizes"]
        if "horizon" in params:
            expected = f"h{int(params['horizon'])}"
            actual = config.get("_horizon", "?")
            if expected != actual:
                print(f"WARNING: hpo_best was tuned for horizon={expected}, running with {actual}")
        print(f">>> HPO params loaded from {hpo_path} (val_loss={hpo.get('val_loss', '?'):.4f})")

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

    loggers = []
    if config["csv_logger"]:
        csv_logger = lgrs.CSVLogger(save_dir="./experiments", name=config["name"])
        loggers.append(csv_logger)
    if config["wandb_logger"]:
        wandb_logger = lgrs.WandbLogger(project="deep_lst", name=config["name"])
        loggers.append(wandb_logger)

    if config["csv_logger"]:
        run_dir = Path(csv_logger.log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config), run_dir / "config.yml")
        metadata = {
            "name": name,
            "config": config["_model_name"],
            "smoke": smoke,
            "tag": tag,
            "seed": config["seed"],
            "git_hash": _git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURM_NODELIST"),
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    torch.set_float32_matmul_precision("medium")

    callbacks = []
    if not smoke and config["csv_logger"]:
        monitor = config["model"].get("checkpoint_monitor", "val_loss_step_epoch")
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
    if config["early_stopping"]:
        callbacks.append(
            EarlyStopping(
                monitor=config["model"].get("checkpoint_monitor", "val_loss_step_epoch_log"),
                mode="min",
                patience=7,
                min_delta=1e-4,
                verbose=True,
            )
        )

    data_module = LitDataModule(**config)
    data_module.setup()

    td_keys = get_keys_from_config(config["train_dataset"])
    packing = get_packing_keys(config["train_dataset"])
    lookback = config["train_dataset"]["seviri"]["lookback"]

    print(f"packing: {packing}")

    inc_variances: dict[str, float] = {}
    stats_tag = stats_spatial(config["train_dataset"])
    for modality in config["modality_ids"]:
        inc_variances.update(load_inc_variances(paths.stats_zarr_path(modality, stats_tag)))

    model = build_model(config, config["train_dataset"], lookback, paths)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"static_encoding={config['model'].get('static_encoding', 'embedding')} parameters={n_parameters:,}")
    if getattr(model, "static_embedder", None) is not None:
        print(f"  {model.static_embedder}")
    model_module = LitModelModule(
        model,
        config,
        **td_keys,
        inc_variances=inc_variances,
        detach_rollout=config["model"].get("detach_rollout", True),
        teacher_forcing_loss=config["model"].get("teacher_forcing_loss", True),
        incremental_loss=config["model"].get("incremental_loss", False),
        rollout_loss=config["model"].get("rollout_loss", True),
    )

    ngpu = torch.cuda.device_count()
    print("number of GPUs:", ngpu)
    if ngpu != 0:
        print("GPU:", torch.cuda.get_device_name(0))
        accelerator = "gpu"
        devices = ngpu
    else:
        accelerator = "cpu"
        devices = 1

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

    start_time = time.time()
    trainer.fit(model_module, datamodule=data_module, ckpt_path=config.get("resume_ckpt"))
    print("Time required for fitting ... minutes ...:", (time.time() - start_time) / 60)

    start_time = time.time()
    test_out = trainer.test(model_module, datamodule=data_module)
    if config["wandb_logger"]:
        test_loss = test_out[0].get("test_loss_step", test_out[0].get("test_loss"))
        if test_loss is not None:
            wandb_logger.experiment.summary["test_loss"] = float(test_loss)
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
