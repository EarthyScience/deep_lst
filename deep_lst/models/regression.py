from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import code

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

        self.step_loss = MaskedMSEStep(
            key_predictions = ("predictions", "LST"),
            key_targets     = ("seviri", "LST")
        )

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
    
    def training_step(self, batch, batch_idx):

        out = self.model(batch)

        y_hat = out['predictions', 'LST']
        y = out['seviri', 'LST']

        valid = torch.isfinite(y) & torch.isfinite(y_hat)
        n_valid = valid.sum()

        if n_valid == 0:
            self.log("train_valid_count", 0, prog_bar=True, sync_dist=True)
            return None 

        loss = F.mse_loss(y_hat[valid], y[valid])

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):  

        out = self.model(batch)

        y_hat = out['predictions','LST']
        y = out['seviri','LST']

        valid = torch.isfinite(y) & torch.isfinite(y_hat)
        n_valid = valid.sum()

        if n_valid == 0:
            self.log("val_valid_count", 0, prog_bar=True, sync_dist=True)
            return None 

        loss = F.mse_loss(y_hat[valid], y[valid])

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in parameter {name}")
        return optimizer
    


def masked_mse(y_hat, y):

    valid = torch.isfinite(y) & torch.isfinite(y_hat)
    n_valid = valid.sum()

    if n_valid == 0:
        return (y_hat.sum() * 0.0), n_valid

    loss = F.mse_loss(y_hat[valid], y[valid])

    return loss, n_valid


def td_num_nonfinite(x) -> int:
    """Return number of non-finite values (NaN or +/-Inf) in tensor x; else 0."""
    return int((~torch.isfinite(x)).sum().item())

def td_num_nan(x) -> int:
    """Return number of NaNs in tensor x; else 0."""
    return int(torch.isnan(x).sum().item())