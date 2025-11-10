from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(LightningModule):

    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU(), dropout=0.1):
        super(MLP, self).__init__()

        self.save_hyperparameters(ignore=['activation'])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.activation = activation
        self.dropout = dropout

        self.num_layers = len(hidden_size) if isinstance(hidden_size, (list, tuple)) else 1

        self.model = self.dense()

    def forward(self, x):
        return self.model(x)
    
    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_size, self.hidden_size[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_size[-1], self.output_size))
        
        return layers
    
    def training_step(self, batch, batch_idx):
        X, y = batch

        y_hat = self.forward(X)

        cloud_mask = torch.isfinite(y)

        y = y[cloud_mask]
        y_hat = y_hat[cloud_mask]

        loss = F.mse_loss(y_hat, y)

        self.log("train_valid_count", cloud_mask.sum(), reduce_fx="sum")
        self.log('train_loss', loss, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):  
        X, y = batch

        y_hat = self.forward(X)

        cloud_mask = torch.isfinite(y)

        y = y[cloud_mask]
        y_hat = y_hat[cloud_mask]

        if cloud_mask.sum() == 0:
            return 0.0

        loss = F.mse_loss(y_hat, y)


        self.log("val_valid_count", cloud_mask.sum(), reduce_fx="sum")
        self.log('val_loss', loss, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    