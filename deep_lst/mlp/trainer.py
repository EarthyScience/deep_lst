import os
import sys

sys.path.append("/Net/Groups/BGI/people/mwessel/deep_lst")  # for deep_lst imports without installation

from deep_lst.misc.utils import open_config
from deep_lst.mlp.datamodule import MLPDataModule
from deep_lst.mlp.model import MLP
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


CSV_LOGGER = loggers.CSVLogger(
        save_dir=f"./experiments/{args.experiment}/logs/",
        name=f"prototype_{args.experiment}"
    )
WANB_LOGGER = loggers.WandbLogger(
        project="deep_lst",
        name=f"prototype_{args.experiment}"
    )

def main():

    config = open_config(experiment=args.experiment, path="./experiments")
    print(config)

    torch.set_float32_matmul_precision('medium')

    data_module = MLPDataModule(config=config, 
                                batch_size=config['batch_size'], 
                                num_workers=16)
    data_module.setup()

    hidden_sizes = config['hidden_sizes']  # Example hidden layer sizes

    model = MLP(input_size=len(data_module.feature_list), 
                hidden_size=hidden_sizes, 
                output_size=len(data_module.target_list))

    ngpu = torch.cuda.device_count() 
    print("GPU:", torch.cuda.get_device_name(0))

    trainer = pl.Trainer(max_epochs=config['max_epochs'], 
                        logger=[CSV_LOGGER, WANB_LOGGER],
                        accelerator="gpu",
                        devices=ngpu,
                        num_nodes=2,
                        strategy="ddp",
                        limit_train_batches=config['train_batches_fraction'],
                        limit_val_batches=config['train_batches_fraction'])

    start_time = time.time()
    print("Fitting the model.")
    trainer.fit(model, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time)/60)

if __name__ == "__main__":
    main()