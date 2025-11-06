import os
print(os.getcwd())
import sys
print(sys.path)
sys.path.append("/Net/Groups/BGI/people/mwessel/deep_lst")  # for deep_lst imports without installation

from deep_lst.misc.utils import open_config
from deep_lst.mlp.datareader import wedataReader, weDataset
from deep_lst.mlp.datamodule import MLPDataModule
from deep_lst.mlp.model import MLP

import pytorch_lightning as pl
import time 

config = open_config("johannisburg")

print(config)

input_dim = 5
output_dim = 1
hidden_dim = 64

model = MLP(input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim)

data_module = MLPDataModule(config=config, 
                            batch_size=256, 
                            num_workers=8)

trainer = pl.Trainer(max_epochs=10, 
                     devices=1)

start_time = time.time()
print("Fitting the model.")
trainer.fit(model, datamodule=data_module)
print("Time required for fitting ... minutes ...:", (time.time() - start_time)/60)

