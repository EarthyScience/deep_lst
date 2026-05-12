"""Hyperparameter optimisation sweep for mlp_rollout_S_KG_crps_lb6_bptt_v3.

Tunes: learning_rate, batch_size, dropout, weight_decay using Optuna TPE.
Uses size=S training data for speed.

Outputs (all written to deep_lst/nas/):
    hpo.db            — Optuna SQLite study (persistent across restarts)
    hpo_results.csv   — All trial results
    hpo_best.json     — Best hyperparameters

Usage:
    python deep_lst/nas/hpo.py                      # 50 trials (default)
    HPO_N_TRIALS=100 python deep_lst/nas/hpo.py     # custom trial count
    sbatch deep_lst/nas/run_hpo.sh                  # via SLURM
"""

import copy
import json
import os
import sys
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.mlp.model import LitModelModule, MultiModalMLP
from deep_lst.utils.config import get_keys_from_config
from deep_lst.utils.io import ExperimentPaths, load_inc_variances

HPO_EPOCHS = 13
HPO_TRAIN_BATCHES = 300
HPO_VAL_BATCHES = 100


def objective(
    trial: optuna.Trial,
    base_config: dict,
    data_modules: dict,
    td_keys: dict,
    in_feature_sizes: dict,
    inc_variances: dict,
) -> float:
    config = copy.deepcopy(base_config)

    config["model"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    config["model"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    horizon = trial.suggest_categorical("horizon", [3, 6, 12])
    print(
        f"[Trial {trial.number}] lr={config['model']['learning_rate']:.2e}  "
        f"dropout={config['model']['dropout']:.2f}  "
        f"wd={config['model']['weight_decay']:.2e}  "
        f"bs={config['training']['batch_size']}  horizon={horizon}",
        flush=True,
    )
    for split in ["train_dataset", "val_dataset"]:
        for modality in ["era5", "seviri"]:
            config[split][modality]["horizon"] = horizon

    config["training"]["max_epochs"] = HPO_EPOCHS
    config["training"]["limit_train_batches"] = HPO_TRAIN_BATCHES
    config["training"]["limit_val_batches"] = HPO_VAL_BATCHES
    config["csv_logger"] = False
    config["wandb_logger"] = False

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
    )

    ngpu = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_epochs=HPO_EPOCHS,
        accelerator="gpu" if ngpu > 0 else "cpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=HPO_TRAIN_BATCHES,
        limit_val_batches=HPO_VAL_BATCHES,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(model_module, datamodule=data_modules[horizon])

    val_loss = trainer.callback_metrics.get("val_loss_step")
    if val_loss is None:
        raise optuna.TrialPruned()
    return float(val_loss)


def main():
    optuna.logging.set_verbosity(optuna.logging.INFO)
    torch.set_float32_matmul_precision("medium")

    with initialize(config_path="../../conf", version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "model=mlp_rollout",
                "loss=crps",
                "stratification=KG",
                "size=L",  # L statistics for correct V3 normalization
                "horizon=h6",
                "lookback=lb6",
                "model.detach_rollout=false",
                "model.scheduler_T0=160000",
            ],
        )
    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("smoke", None)
    config.pop("tag", None)

    # Build one data module per horizon — reused across all trials for that horizon
    data_modules: dict[int, LitDataModule] = {}
    for h in [3, 6, 12]:
        cfg_h = copy.deepcopy(config)
        for split in ["train_dataset", "val_dataset"]:
            for modality in ["era5", "seviri"]:
                cfg_h[split][modality]["horizon"] = h
        dm = LitDataModule(**cfg_h)
        dm.setup()
        data_modules[h] = dm

    # td_keys and inc_variances are horizon-independent
    td_keys = get_keys_from_config(config["train_dataset"])
    lookback = config["train_dataset"]["seviri"]["lookback"]
    context_vars = config["train_dataset"]["seviri"].get("context_vars", [])
    in_feature_sizes: dict = {}
    if lookback > 0:
        for var in context_vars:
            ctx_key = ("seviri", f"{var}_ctx")
            td_keys["in_keys"].append(ctx_key)
            in_feature_sizes[ctx_key] = 2 * lookback

    paths = ExperimentPaths()
    inc_variances: dict[str, float] = {}
    for modality, mod_cfg in config["train_dataset"].items():
        if isinstance(mod_cfg, dict) and "statistics" in mod_cfg:
            inc_variances.update(load_inc_variances(paths.statistics_path(modality, mod_cfg["statistics"])))

    nas_dir = Path(__file__).parent.parent.parent / "nas"
    nas_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{nas_dir}/hpo.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage,
        study_name="mlp_rollout_S_KG_crps_lb6_bptt_v3_hpo",
        load_if_exists=True,
    )

    n_trials = int(os.environ.get("HPO_N_TRIALS", "100"))
    study.optimize(
        lambda trial: objective(trial, config, data_modules, td_keys, in_feature_sizes, inc_variances),
        n_trials=n_trials,
    )

    df = study.trials_dataframe()
    df.to_csv(nas_dir / "hpo_results.csv", index=False)

    best = study.best_trial
    with open(nas_dir / "hpo_best.json", "w") as f:
        json.dump({"val_loss": best.value, "params": best.params}, f, indent=2)

    print("\nBest trial:")
    print(f"  val_loss: {best.value:.6f}")
    print(f"  params:   {best.params}")
    print(f"\nResults saved to {nas_dir}/")


if __name__ == "__main__":
    main()
