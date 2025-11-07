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

CSV_LOGGER = loggers.CSVLogger(
        save_dir="./experiments/johannisburg/logs/",
        name="mlp_logs"
    )
WANB_LOGGER = loggers.WandbLogger(
        project="deep_lst",
        name="mlp_johannisburg"
    )

def main():

    config = open_config(experiment="johannisburg", path="./experiments")
    print(config)

    torch.set_float32_matmul_precision('medium')

    data_module = MLPDataModule(config=config, 
                                batch_size=256, 
                                num_workers=30)
    data_module.setup()

    hidden_sizes = [128, 64, 128]  # Example hidden layer sizes

    model = MLP(input_size=len(data_module.feature_list), 
                hidden_size=hidden_sizes, 
                output_size=len(data_module.target_list))

    use_gpu = torch.cuda.is_available()
    ngpu = torch.cuda.device_count() 
    print("GPU:", torch.cuda.get_device_name(0))

    trainer = pl.Trainer(max_epochs=10, 
                        logger=[CSV_LOGGER, WANB_LOGGER],
                        accelerator="gpu",
                        devices=ngpu,
                        strategy="ddp")

    start_time = time.time()
    print("Fitting the model.")
    trainer.fit(model, datamodule=data_module)
    print("Time required for fitting ... minutes ...:", (time.time() - start_time)/60)

if __name__ == "__main__":
    main()