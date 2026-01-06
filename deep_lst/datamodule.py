import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from deep_lst.dataset import UnimodalDataset, MultimodalDataset
from tensordict import TensorDict
from tensordict import stack as td_stack

def variable_keys(config, modality, var_type: str) -> list[tuple[str, str]]:
    keys = []
    for var in config['modality'][modality].get('variables', []):
        (varname, meta), = var.items()
        if meta.get("type") == var_type:
            keys.append((modality, varname))
    return keys

class DataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.modality_ids = list(sorted(config['modality'].keys())) # era, seviri

        self._assemble_features()

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
        
        self.input_size = sum(len(var) for var in self.inputs.values())
        self.target_size = sum(len(var) for var in self.targets.values())

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_multimodal_tensordict
        )
    
    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,  
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_multimodal_tensordict
        )

    def _assemble_features(self):

        """
        assembles an dict of input and target feature key set.
        sets class attributes self.inputs and self.targets.
        """
        self.inputs = {mod : variable_keys(self.config, modality=mod, var_type="input") for mod in self.modality_ids}
        self.targets = {mod : variable_keys(self.config, modality=mod, var_type="diagnostic") for mod in self.modality_ids}

    def collate_multimodal_tensordict(self, batch, pack_tensors=True):
        """
        batch: list of samples
        each sample is dict: {modality_id: TensorDict}
        returns:
        dict: {modality_id: TensorDict with batch dim prepended}
        
        if pack_tensors, returns a TensorDict with additional inputs key that contains a tensor per modality stacked along channels.
        """
        stacked = {modality: td_stack([sample[modality] for sample in batch], dim=0) for modality in self.modality_ids}

        td_out = TensorDict({}, batch_size=self.config['batch_size'], device=self.config['device'])
        # Put each modality under a nested key namespace
        for modality in self.modality_ids:
            for key, values in stacked[modality].items():
                td_out.set((modality, key), values)

        if not pack_tensors:
            return td_out # returns nested TensorDict

        for modality in self.modality_ids:
            # Packs all channels in a modality into one tensor at TensorDict['inputs'] with channel dimension at -1
            in_keys = self.inputs.get(modality, [])
            if in_keys:
                td_out.set(("inputs", modality),
                        torch.stack([td_out.get(k) for k in in_keys], dim=-1))

            tgt_keys = self.targets.get(modality, [])
            if tgt_keys:
                td_out.set(("targets", modality),
                        torch.stack([td_out.get(k) for k in tgt_keys], dim=-1))

        return td_out

