import importlib

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule

# from deep_lst.deep_lst.mlp.loss_logic import DiagnosticMSE, RolloutMSE
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset


class LitDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.datareader_pars = config["datareader"]
        self.training_pars = config["training"]
        self.collate_fn = Collator(config["model"]["name"])

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
            batch_size=self.training_pars["batch_size"],
            shuffle=True,
            num_workers=self.training_pars["num_workers"],
            pin_memory=self.training_pars["pin_memory"],
            prefetch_factor=self.training_pars["prefetch_factor"],
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_pars["batch_size"],
            shuffle=True,
            num_workers=self.training_pars["num_workers"],
            pin_memory=self.training_pars["pin_memory"],
            prefetch_factor=self.training_pars["prefetch_factor"],
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_pars["batch_size"],
            shuffle=True,
            num_workers=self.training_pars["num_workers"],
            pin_memory=self.training_pars["pin_memory"],
            prefetch_factor=self.training_pars["prefetch_factor"],
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )


class LitModelModule(LightningModule):
    def __init__(self, model, config):
        super().__init__()

        self.model = model
        self.config = config

        self.step_loss = self._get_loss(model_id=config["name"], loss_name=config["step_loss"])()

    def _get_loss(self, model_id: str, loss_name: str):
        module = importlib.import_module(f"deep_lst.{model_id}")
        return getattr(module, loss_name)

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learning_rate"],
            eps=1e-8,  # Numerical stability
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.get("max_epochs", 25))

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_train_epoch_end(self):
        """Clear any accumulated state at epoch end"""
        torch.cuda.empty_cache()
