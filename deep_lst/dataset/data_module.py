import pytorch_lightning as pl
from torch.utils.data import DataLoader

from deep_lst.dataset.collator import Collator
from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.utils.config import get_packing_keys


class LitDataModule(pl.LightningDataModule):
    def __init__(self, experiment_paths=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.paths = experiment_paths
        self.collate_fn = Collator(kwargs["model"]["name"], get_packing_keys(kwargs["train_dataset"]))

    def setup(self, stage=None):
        if hasattr(self, "train_dataset"):
            return

        # Preload the pixel columns once per spatial selection and share the RAM cache
        # between the splits on it; val/test may hold out different pixels than train.
        self._shared_by_spatial = {}
        self.train_dataset = self._build_split(self.kwargs["train_dataset"])
        self.val_dataset = self._build_split(self.kwargs["val_dataset"])
        self.test_dataset = self._build_split(self.kwargs["test_dataset"])

    def _build_split(self, split_kwargs):
        modality_ids = self.kwargs["modality_ids"]
        spatial = split_kwargs["spatial"]
        if spatial not in self._shared_by_spatial:
            self._shared_by_spatial[spatial] = MultimodalDataset.preload_shared(
                modality_ids, paths=self.paths, **split_kwargs
            )
        shared = self._shared_by_spatial[spatial]
        return MultimodalDataset.from_modalities(modality_ids, shared, paths=self.paths, **split_kwargs)

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
