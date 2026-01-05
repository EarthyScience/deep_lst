import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from deep_lst.dataset import UnimodalDataset, MultimodalDataset

class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']

    def setup(self, stage=None):

        self.train_dataset = MultimodalDataset.from_modalities(
            modality_ids=["era5", "seviri"],
            config=self.config,
            split="train",
        )
        self.val_dataset = MultimodalDataset.from_modalities(
            modality_ids=["era5", "seviri"],
            config=self.config,
            split="val",
        )

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
    
    def collate_multimodal_tensordict(batch):
        """
        batch: list of samples
        each sample is dict: {modality_id: TensorDict}
        returns:
        dict: {modality_id: TensorDict with batch dim prepended}
        """
        modalities = batch[0].keys()
        out = {}
        for m in modalities:
            tds = [sample[m] for sample in batch]
            out[m] = td_stack(tds, dim=0)  # batch dimension
        return out