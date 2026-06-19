"""Hyperparameter optimisation sweep."""

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
from deep_lst.utils.config import get_context_keys, get_keys_from_config
from deep_lst.utils.io import ExperimentPaths, load_inc_variances

HPO_EPOCHS = 20
# Per-epoch SAMPLE budgets (not batch counts). limit_train/val_batches are derived per trial as
# SAMPLES // batch_size, so every batch size sees the same data — a fair batch-size comparison.
# Batch sizes are floored at 32 (bs=32 → 625 train batches/epoch) so the smallest-bs trials stay
# feasible; below that the equalised-sample scheme makes a single trial take 13–37 h.
HPO_TRAIN_SAMPLES = 20000
HPO_VAL_SAMPLES = 8000
VAL_METRIC = "val_loss_step_epoch"

HIDDEN_SIZES = [
    [512, 256, 256, 128, 256],
    [1024, 512, 256, 256, 128],
    [1024, 512, 512, 256, 256],
    [2048, 1024, 512, 512, 256],
    [4096, 2048, 1024, 1024, 512],
]


class _PruningCallback(pl.Callback):
    """Report the validation metric to Optuna each epoch and prune unpromising trials early."""

    def __init__(self, trial: optuna.Trial, monitor: str):
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), trainer.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial {self.trial.number} pruned at epoch {trainer.current_epoch}")


def objective(
    trial: optuna.Trial,
    base_config: dict,
    data_module: LitDataModule,
    td_keys: dict,
    in_feature_sizes: dict,
    inc_variances: dict,
) -> float:
    config = copy.deepcopy(base_config)

    config["model"]["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
    config["model"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config["model"]["hidden_sizes"] = HIDDEN_SIZES[trial.suggest_categorical("hidden_sizes_idx", [0, 1, 2, 3, 4])]
    config["training"]["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    print(
        f"[Trial {trial.number}] lr={config['model']['learning_rate']:.2e}  "
        f"dropout={config['model']['dropout']:.2f}  "
        f"wd={config['model']['weight_decay']:.2e}  "
        f"bs={config['training']['batch_size']}",
        flush=True,
    )

    bs = config["training"]["batch_size"]
    config["training"]["max_epochs"] = HPO_EPOCHS
    # Equalise data seen across batch sizes: fixed SAMPLES/epoch → #batches = SAMPLES // bs.
    config["training"]["limit_train_batches"] = max(1, HPO_TRAIN_SAMPLES // bs)
    config["training"]["limit_val_batches"] = max(1, HPO_VAL_SAMPLES // bs)
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
        detach_rollout=config["model"].get("detach_rollout", False),
        incremental_loss=config["model"].get("incremental_loss", True),
    )

    ngpu = torch.cuda.device_count()
    trainer = pl.Trainer(
        max_epochs=HPO_EPOCHS,
        accelerator="gpu" if ngpu > 0 else "cpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        limit_train_batches=config["training"]["limit_train_batches"],
        limit_val_batches=config["training"]["limit_val_batches"],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        callbacks=[_PruningCallback(trial, VAL_METRIC)],
    )
    trainer.fit(model_module, datamodule=data_module)

    val_loss = trainer.callback_metrics.get(VAL_METRIC)
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
                "loss=smoothl1",
                "stratification=mtg",
                "size=L",
                "horizon=h24",
                "model.warmup_steps=300",
            ],
        )
    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("smoke", None)
    config.pop("tag", None)

    dm = LitDataModule(**config)
    dm.setup()

    # td_keys and inc_variances are horizon-independent
    td_keys = get_keys_from_config(config["train_dataset"])
    lookback = config["train_dataset"]["seviri"]["lookback"]
    in_feature_sizes: dict = {}
    if lookback > 0:
        for ctx_key, size in get_context_keys(config["train_dataset"], lookback):
            td_keys["in_keys"].append(ctx_key)
            in_feature_sizes[ctx_key] = size

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
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
        storage=storage,
        # New study name: the batch_size categorical changed (8/16 dropped), which is incompatible
        # with resuming the old study. The old study's trials remain in hpo.db.
        study_name="mlp_rollout_MT_smoothl1_h24_l24_v6_hpo_bs32",
        load_if_exists=True,
    )

    n_trials = int(os.environ.get("HPO_N_TRIALS", "25"))
    study.optimize(
        lambda trial: objective(trial, config, dm, td_keys, in_feature_sizes, inc_variances),
        n_trials=n_trials,
    )

    df = study.trials_dataframe()
    df.to_csv(nas_dir / "hpo_results.csv", index=False)

    best = study.best_trial
    best_hidden_sizes = HIDDEN_SIZES[best.params["hidden_sizes_idx"]]
    with open(nas_dir / "hpo_best.json", "w") as f:
        json.dump({"val_loss": best.value, "params": best.params, "hidden_sizes": best_hidden_sizes}, f, indent=2)

    print("\nBest trial:")
    print(f"  val_loss: {best.value:.6f}")
    print(f"  params:   {best.params}")
    print(f"\nResults saved to {nas_dir}/")


if __name__ == "__main__":
    main()
