import os
import sys

sys.path.append("/Net/Groups/BGI/people/mwessel/deep_lst")  # for deep_lst imports without installation

from deep_lst.misc.utils import open_config
from deep_lst.datamodule import DataModule
from deep_lst.models.regression import MLP
from pytorch_lightning import loggers
from lightning_fabric.plugins.environments import LightningEnvironment

import pytorch_lightning as pl
import time 
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", required=True, type=str, help="Name of the experiment.")
args = parser.parse_args()

print("SLURM_NODELIST:", os.environ.get("SLURM_NODELIST"))
print("SLURM_NNODES:", os.environ.get("SLURM_NNODES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS")) 

def get_feature_lengths(config: dict) -> tuple[int, int]:
    """
    Returns (n_input_features, n_target_features) by counting variables with
    spec['type'] == 'input' / 'target' across all modalities.
    """
    input_vars = target_vars = 0
    for mod in config.get("modality", {}).values():
        for item in mod.get("variables", []):
            for _, spec in item.items():
                typ = (spec or {}).get("type")
                if typ == "input":
                    input_vars += 1
                elif typ == "target":
                    target_vars += 1
    return input_vars, target_vars

def main():

    config = open_config(experiment="test", path="experiments")

    csv_logger = loggers.CSVLogger(
        save_dir=f"./experiments/{args.experiment}/logs/",
        name=config['name']
    )
    wandb_logger = loggers.WandbLogger(
            project="deep_lst",
            name=config['name']
        )

    torch.set_float32_matmul_precision('medium')

    data_module = DataModule(config=config)
    data_module.setup()
    
    input_size, output_size = get_feature_lengths(config)

    model = MLP(input_size=input_size, 
                hidden_size=config['hidden_sizes'] , 
                output_size=output_size)

    ngpu = torch.cuda.device_count() 
    print("GPU:", torch.cuda.get_device_name(0))

    trainer = pl.Trainer(max_epochs=config['max_epochs'], 
                        logger=[csv_logger, wandb_logger],
                        accelerator="gpu",
                        devices=ngpu,
                        num_nodes=2,
                        strategy="ddp",
                        limit_train_batches=0.3,
                        limit_val_batches=0.3)

    start_time = time.time()
    print("Fitting the model.")
    trainer.fit(model, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time)/60)

if __name__ == "__main__":
    main()