import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from deep_lst.dataset.dataset import UnimodalDataset, MultimodalDataset
from tensordict import TensorDict
from tensordict import stack as td_stack


class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.datareader_pars = config['datareader']
        self.training_pars = config["training"]

    def setup(self, stage=None):

        self.train_dataset = MultimodalDataset.from_modalities(
            config=self.datareader_pars,
            split="train",
        )
        self.val_dataset = MultimodalDataset.from_modalities(
            config=self.datareader_pars,
            split="val",
        )

        self.test_dataset = MultimodalDataset.from_modalities(
            config=self.datareader_pars,
            split="test",
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.training_pars['batch_size'],
            shuffle=True,
            num_workers=self.training_pars['num_workers'],
            pin_memory=self.training_pars['pin_memory'],
            prefetch_factor=self.training_pars ['prefetch_factor'],
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context='fork',
            collate_fn=self.collate_multimodal_tensordict
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.training_pars['batch_size'],
            shuffle=True,
            num_workers=self.training_pars['num_workers'],
            pin_memory=self.training_pars['pin_memory'],
            prefetch_factor=self.training_pars ['prefetch_factor'],
            drop_last=True,
            multiprocessing_context='fork',
            collate_fn=self.collate_multimodal_tensordict
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.training_pars['batch_size'],
            shuffle=True,
            num_workers=self.training_pars['num_workers'],
            pin_memory=self.training_pars['pin_memory'],
            prefetch_factor=self.training_pars ['prefetch_factor'],
            drop_last=True,
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

