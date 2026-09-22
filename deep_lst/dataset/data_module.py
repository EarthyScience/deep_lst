import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.utils.config import get_packing_keys


class LitDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.collate_fn = Collator(kwargs["model"]["name"], get_packing_keys(kwargs["train_dataset"]))

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        # Preload the pixel columns once (same pixels/stores for every split) and share
        # the RAM cache across train/val/test; only the temporal anchors differ.
        modality_ids = self.kwargs["modality_ids"]
        shared = MultimodalDataset.preload_shared(modality_ids, **self.kwargs["train_dataset"])

        self.train_dataset = MultimodalDataset.from_modalities(modality_ids, shared, **self.kwargs["train_dataset"])
        self.val_dataset = MultimodalDataset.from_modalities(modality_ids, shared, **self.kwargs["val_dataset"])
        self.test_dataset = MultimodalDataset.from_modalities(modality_ids, shared, **self.kwargs["test_dataset"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.kwargs["training"]["batch_size"],
            shuffle=True,
            num_workers=self.kwargs["training"]["num_workers"],
            pin_memory=self.kwargs["training"]["pin_memory"],
            prefetch_factor=self.kwargs["training"]["prefetch_factor"],
            persistent_workers=True,
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.kwargs["training"]["batch_size"],
            shuffle=False,
            num_workers=self.kwargs["training"]["num_workers"],
            pin_memory=self.kwargs["training"]["pin_memory"],
            prefetch_factor=self.kwargs["training"]["prefetch_factor"],
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.kwargs["training"]["batch_size"],
            shuffle=False,
            num_workers=self.kwargs["training"]["num_workers"],
            pin_memory=self.kwargs["training"]["pin_memory"],
            prefetch_factor=self.kwargs["training"]["prefetch_factor"],
            drop_last=True,
            multiprocessing_context="fork",
            collate_fn=self.collate_fn,
        )
