import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from deep_lst.mlp.datareader import wedataReader, weDataset

class MLPDataModule(pl.LightningDataModule):

    def __init__(self, config, batch_size: int = 256, num_workers: int = 8):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        self.train_dataset = weDataset(config=self.config, split='train')
        self.val_dataset = weDataset(config=self.config, split='val')
        self.test_dataset = weDataset(config=self.config, split='test')

        self.feature_list = self.train_dataset.feature_list
        self.target_list = ['LST']  

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size*4,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,  
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,  
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        inputs_list, targets_list = zip(*batch) 
        
        X_rows = []
        for inp in inputs_list:
            # flatten scalars/1D to 1D and concatenate in your desired key order
            row = torch.cat([inp[k].float().reshape(-1) for k in self.feature_list], dim=0)
            X_rows.append(row)
        X = torch.stack(X_rows, dim=0)  # [B, F]

        y_rows = []
        for tgt in targets_list:
            y_row = torch.cat([tgt[k].float().reshape(-1) for k in self.target_list], dim=0)
            y_rows.append(y_row)
        y = torch.stack(y_rows, dim=0)
        return X, y