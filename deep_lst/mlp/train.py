"""
To test with resolved config: python deep_lst/mlp/train.py loss=crps horizon=h6 --cfg job --resolve
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
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils.config import get_context_keys, get_keys_from_config
from deep_lst.utils.io import ExperimentPaths, load_inc_variances

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


@hydra.main(config_path="../../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42, workers=True)

    paths = ExperimentPaths()

    config = OmegaConf.to_container(cfg, resolve=True)

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

    # Build run name from model + experiment keys set by config groups
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

    # --- snapshot config + metadata into the experiment run directory ---
    if config["csv_logger"]:
        run_dir = Path(csv_logger.log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(config), run_dir / "config.yml")
        metadata = {
            "name": name,
            "config": config["_model_name"],
            "smoke": smoke,
            "tag": tag,
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
                monitor=config["model"].get("checkpoint_monitor", "val_loss_step_epoch"),
                mode="min",
                patience=7,
                min_delta=1e-4,
                verbose=True,
            )
        )

    data_module = LitDataModule(**config)
    data_module.setup()

    climatology = None
    if config["model"].get("predict_anomaly", False):
        from eodata.converter.variables import STORE_TIME_RANGE

        from deep_lst.dataset.climatology import ClimatologyLookup

        spatial = config["train_dataset"]["spatial"]
        lst_norm = data_module.train_dataset.readers["seviri"].normalizers["LST"]

        climatology = ClimatologyLookup(
            clim_path=paths.climatology_path(f"train_{spatial}", "smoothed_hourly"),
            spatial_idx_path=paths.spatial_indices_path("seviri", spatial),
            lookback=config["train_dataset"]["seviri"]["lookback"],
            horizon=config["train_dataset"]["seviri"]["horizon"],
            state_mean=float(lst_norm.mean),
            state_std=float(lst_norm.std),
            store_time_range=STORE_TIME_RANGE,
            variable="LST",
        )

    td_keys = get_keys_from_config(config["train_dataset"])

    lookback = config["train_dataset"]["seviri"]["lookback"]
    in_feature_sizes: dict = {}
    if lookback > 0:
        for ctx_key, size in get_context_keys(config["train_dataset"], lookback):
            td_keys["in_keys"].append(ctx_key)
            in_feature_sizes[ctx_key] = size

    print(f"in_keys: {td_keys['in_keys']}")
    print(f"in_feature_sizes: {in_feature_sizes}")

    inc_variances: dict[str, float] = {}
    for modality, mod_cfg in config["train_dataset"].items():
        if isinstance(mod_cfg, dict) and "statistics" in mod_cfg:
            inc_variances.update(load_inc_variances(paths.statistics_path(modality, mod_cfg["statistics"])))

    model = MultiModalMLP(
        in_keys=td_keys["in_keys"] + td_keys["prognostic_keys"],
        out_keys=td_keys["diagnostic_out_keys"] + td_keys["prognostic_out_keys"],
        config=config["model"],
        in_feature_sizes=in_feature_sizes,
    )
    model_module = LitModelModule(
        model,
        config,
        **td_keys,
        inc_variances=inc_variances,
        detach_rollout=config["model"].get("detach_rollout", True),
        incremental_loss=config["model"].get("incremental_loss", False),
        climatology=climatology,
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
    trainer.fit(model_module, datamodule=data_module)
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
