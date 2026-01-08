import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

from pathlib import Path
from deep_lst.utils.time_filter import get_temporal_indices
from wedata.eo_zarr import EOZarrStore

from tensordict import TensorDict
from tensordict import stack as td_stack

def get_reader_from_modality(modality_id: str, config: dict, split: str):
    if modality_id == "era5":
        from wedata.data_reader.era5 import Era5Reader as Reader
    elif modality_id == "seviri":
        from wedata.data_reader.seviri import SeviriReader as Reader
    else:
        raise ValueError(f"Unknown modality id {modality_id}")

    store_path = Path(config["store_path"]) / config["modality"][modality_id]["store_name"]
    indices_path = Path(config["assets_path"]) / modality_id / config["experiment"] / "indices" / "indices.npz"
    statistics_path = Path(config["assets_path"]) / modality_id / config["experiment"] / "statistics" / "statistics.npz"

    store = EOZarrStore(store_path, mode="r+")

    spatial_idx = np.load(indices_path)["mapping"]
    temporal_idx = get_temporal_indices(config, split=split, stratify=False)

    reader = Reader(eozarr_store=store, config=config)

    reader.available_indices(indices=spatial_idx)
    reader.set_normalizer(file=statistics_path)

    return reader, spatial_idx, temporal_idx

def variable_keys(config, modality, var_type: str) -> list[tuple[str, str]]:
    keys = []
    for var in config['modality'][modality].get('variables', []):
        (varname, meta), = var.items()
        if meta.get("type") == var_type:
            keys.append((modality, varname))
    return keys

class UnimodalDataset(Dataset):
    def __init__(self, reader, spatial_indices, temporal_indices, config, modality_id: str):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.spatial_indices = spatial_indices
        self.reader = reader
        self.temporal_indices = temporal_indices
        self.config = config
        self.modality_id = modality_id
        
        self.P = len(self.spatial_indices)
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        self.inputs = {mod : variable_keys(self.config, modality=mod, var_type="input") for mod in self.modality_ids}
        self.targets = {mod : variable_keys(self.config, modality=mod, var_type="diagnostic") for mod in self.modality_ids}

        print(f"Initialized Unimodal dataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = self.temporal_indices[idx // self.P]
        p_index = self.spatial_indices[idx % self.P]

        sample = self.reader.get_data_single(space_idx=p_index, time_idx=t_index)

        return sample
    
    @classmethod
    def from_modality(cls,
        modality_id: str,
        config: dict,
        split: str,
    ):
        reader, spatial_idx, temporal_idx = get_reader_from_modality(modality_id, config, split)

        dataset = cls(
            reader=reader,
            spatial_indices=spatial_idx,
            temporal_indices=temporal_idx,
            config=config,
            modality_id=modality_id,
        )

        return dataset

class MultimodalDataset(Dataset):

    def __init__(self, reader: dict[str], spatial_indices: dict[str], temporal_indices, config):

        self.readers = reader
        self.spatial_indices = spatial_indices
        self.temporal_indices = temporal_indices
        self.modality_ids = list(self.readers.keys())
        self.config = config

        self.P = len(self.spatial_indices[self.modality_ids[0]])
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        self.inputs = {mod : variable_keys(self.config, modality=mod, var_type="input") for mod in self.modality_ids}
        self.targets = {mod : variable_keys(self.config, modality=mod, var_type="diagnostic") for mod in self.modality_ids}

        print(f"Initialized MultimodalDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        
        t_index = self.temporal_indices[idx // self.P]

        data = TensorDict({}, batch_size=[]) 
        for modality_id, reader in self.readers.items():
            p_index = self.spatial_indices[modality_id][idx % self.P]

            data_mod = reader.get_data_single(space_idx=p_index, time_idx=t_index)

            data.set((modality_id,), data_mod)

            input_keys = self.inputs.get(modality_id, [])
            if input_keys:
                data.set(("inputs", modality_id),
                        torch.stack([data.get(k) for k in input_keys], dim=-1))

            target_keys = self.targets.get(modality_id, [])
            if target_keys:
                data.set(("targets", modality_id),
                        torch.stack([data.get(k) for k in target_keys], dim=-1))

        return data

    @classmethod
    def from_modalities(cls,
        modality_ids: list[str],
        config: dict,
        split: str
    ):
        readers = {}
        spatial_indices = {}

        for modality_id in modality_ids:
            reader, spatial_idx, temporal_idx = get_reader_from_modality(modality_id, config, split)
            readers[modality_id] = reader
            spatial_indices[modality_id] = spatial_idx

        dataset = cls(
            reader=readers,
            spatial_indices=spatial_indices,
            temporal_indices=temporal_idx,
            config=config,
        )

        return dataset