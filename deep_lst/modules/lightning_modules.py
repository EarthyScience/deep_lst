import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch

from deep_lst.losses.mse import DiagnosticMSE, RolloutMSE
from torch.utils.data import DataLoader
from deep_lst.dataset.dataset import MultimodalDataset
from tensordict import stack as td_stack


class LitDataModule(pl.LightningDataModule):

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



class LitModelModule(LightningModule):

    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config

        if config['step_loss'] == 'rollout':
            self.step_loss = RolloutMSE()
        elif config['step_loss'] == 'stepwise':
            self.step_loss = DiagnosticMSE()
        else:
            raise ValueError(f"Unknown step_loss type {config['step_loss']}")
    
    def training_step(self, batch, batch_idx):
        loss = self.step_loss(self, batch, stage="train")
        return loss
    
    def validation_step(self, batch, batch_idx):  
        loss = self.step_loss(self, batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):  
        loss = self.step_loss(self, batch, stage="test")
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=self.config['learning_rate'],
                                    eps=1e-8,         # Numerical stability
                                    weight_decay=0.0)
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
