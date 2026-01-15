import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from deep_lst.dataset.dataset import UnimodalDataset, MultimodalDataset
from tensordict import TensorDict
from tensordict import stack as td_stack


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.modality_ids = list(sorted(config['modality'].keys())) # era, seviri

    def setup(self, stage=None):

        self.train_dataset = MultimodalDataset.from_modalities(
            modality_ids=self.modality_ids,
            config=self.config,
            split="train",
        )
        self.val_dataset = MultimodalDataset.from_modalities(
            modality_ids=self.modality_ids,
            config=self.config,
            split="val",
        )
        
        self.input_size = sum(len(var) for var in self.train_dataset.inputs.values())
        self.diagnostic_size = sum(len(var) for var in self.train_dataset.diagnostics.values())
        self.prognostic_size = sum(len(var) for var in self.train_dataset.prognostics.values())

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context='fork',
            collate_fn=self.collate_multimodal_tensordict
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,  
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            drop_last=False,
            multiprocessing_context='fork',
            collate_fn=self.collate_multimodal_tensordict
        )

    def collate_multimodal_tensordict(self, batch, pack_tensors=True):
        """
        batch: list of samples
        each sample is dict: {modality_id: TensorDict}
        returns:
        dict: {modality_id: TensorDict with batch dim prepended}
        
        if pack_tensors, returns a TensorDict with additional inputs key that contains a tensor per modality stacked along channels.
        """
        return td_stack(batch, dim=0) #td_out

