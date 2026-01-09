from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import code
import time
import torch.cuda as cuda

from deep_lst.losses.mse import MaskedMSEStep, MaskedMSERollout

class MultiModalMLP(nn.Module):

    def __init__(self, input_size, target_size, config):
        super().__init__()

        self.input_size = input_size
        self.target_size = target_size

        self.activation = nn.ReLU() if config['activation'] == 'relu' else None
        self.hidden_sizes = config['hidden_sizes']
        self.dropout = config['dropout']
        self.modality_ids = list(config['modality'].keys())

        self.net = self.dense()

    def forward(self, *inputs):
        
        inputs = [x for x in inputs if x is not None]
        x = torch.cat(inputs, dim=-1) # assumes channel is last dim

        return self.net(x)

    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_size, self.hidden_sizes[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if len(self.hidden_sizes) > 1:
            for i in range(len(self.hidden_sizes)-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_sizes[-1], self.target_size))
        return layers


class LitTDmodule(LightningModule):

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config

        self.step_loss = MaskedMSEStep(
            predictions = ("predictions", "LST"),
            targets     = ("seviri", "LST")
        )
    
    def training_step(self, batch, batch_idx):

        loss = self.step_loss(self, batch, stage="train")

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        return loss
    
    def validation_step(self, batch, batch_idx):  

        loss = self.step_loss(self, batch, stage="val")

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.get('max_epochs', 25)
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

    def on_train_epoch_end(self):
        """Clear any accumulated state at epoch end"""
        torch.cuda.empty_cache()
