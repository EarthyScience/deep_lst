import os
import sys

from deep_lst.utils.misc import open_config
from deep_lst.datamodule import DataModule
from deep_lst.models.regression import MultiModalMLP, LitTDmodule
from pytorch_lightning import loggers
from lightning_fabric.plugins.environments import LightningEnvironment
from tensordict.nn import TensorDictModule, TensorDictSequential
from pytorch_lightning.profilers import AdvancedProfiler

import pytorch_lightning as pl
import time 
import torch

import argparse
import code

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", required=True, type=str, help="Name of the experiment.")
args = parser.parse_args()

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS")) 

def keys_by_type_from_cfg(config, var_type: str) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for modality, modality_conf in config["modality"].items():
        for var in modality_conf.get("variables", []):
            (varname, meta), = var.items()
            if meta.get("type") == var_type:
                keys.append((modality, varname))
    return keys

def main():

    config = open_config(experiment="test", path="experiments")

    csv_logger = loggers.CSVLogger(
        save_dir=f"./experiments/{args.experiment_name}/logs/",
        name=config['name']
    )
    wandb_logger = loggers.WandbLogger(
            project="deep_lst",
            name=config['name']
        )

    profiler = AdvancedProfiler(dirpath=f"./experiments/{args.experiment_name}", filename="perf_logs")

    torch.set_float32_matmul_precision('medium')

    data_module = DataModule(config=config)
    data_module.setup()

    td_model = TensorDictModule(
        module=MultiModalMLP(data_module.input_size+data_module.prognostic_size, data_module.diagnostic_size+data_module.prognostic_size, config),
        in_keys=[("inputs", "era5"), ("inputs", "seviri"), ("prognostics", "seviri")],
        out_keys=[("predictions", "LST")],
    )
    lit_model = LitTDmodule(td_model, config)

    ngpu = torch.cuda.device_count() 
    print("GPU:", torch.cuda.get_device_name(0))
    print("number of GPUs:", ngpu)

    trainer = pl.Trainer(max_epochs=config['epochs'], 
                        logger=[csv_logger, wandb_logger],
                        accelerator="gpu",
                        devices=ngpu,
                        num_nodes=1,
                        strategy="ddp",
                        limit_train_batches=config['limit_train_batches'],
                        limit_val_batches=config['limit_val_batches'],
                        precision="bf16-mixed",
                        gradient_clip_val=1.0,
                        gradient_clip_algorithm="norm",
                        log_every_n_steps=50)

    print("Trainer world_size:", trainer.world_size, "devices:", torch.cuda.device_count())

    start_time = time.time()
    print("Fitting the model.")
    trainer.fit(lit_model, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time)/60)

if __name__ == "__main__":
    main()