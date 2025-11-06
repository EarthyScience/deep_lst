import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deep_lst.mlp.datareader import wedataReader, weDataset

class MLPDataModule(pl.LightningDataModule):

    def __init__(self, config, batch_size: int = 256, num_workers: int = 8):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        self.train_dataset = weDataset(
            config=self.config,
            split='train'
        )

        self.val_dataset = weDataset(
            config=self.config,
            split='val'
        )

        self.test_dataset = weDataset(
            config=self.config,
            split='test'
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
