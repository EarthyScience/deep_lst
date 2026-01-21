import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import time

from pathlib import Path
from deep_lst.utils.time_filter import get_temporal_indices
from wedata.eo_zarr import EOZarrStore
import wedata

from tensordict import TensorDict
from tensordict import stack as td_stack

def get_reader_from_modality(modality_id: str, config: dict, split: str):


    store = EOZarrStore(config[modality_id]["store_path"], mode="r+")

    spatial_idx = np.load(config[modality_id]["indices_path"])["mapping"]
    temporal_idx = get_temporal_indices(config, split=split, stratify=False)

    reader = wedata.get_data_reader(modality_id, eozarr_store=store, **config[modality_id])

    reader.available_indices(indices=spatial_idx)
    reader.set_normalizer(file=config[modality_id]["statistics_path"])

    return reader, spatial_idx, temporal_idx


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
        
        self.P = len(self.spatial_indices)
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        self.inputs = config[modality_id]['inputs']
        self.diagnostics = config[modality_id]['diagnostics']
        self.prognostics = config[modality_id]['prognostics']

        print(f"Initialized Unimodal dataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = self.temporal_indices[idx // self.P]
        p_index = self.spatial_indices[idx % self.P]

        sample = self.reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

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

        self.P = len(self.spatial_indices[self.modality_ids[0]])
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        self.inputs = {mod : config[mod]['inputs'] for mod in self.modality_ids}
        self.diagnostics = {mod : config[mod]['diagnostics'] for mod in self.modality_ids}
        self.prognostics = {mod : config[mod]['prognostics'] for mod in self.modality_ids}

        print(f"Initialized MultimodalDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        
        t_index = self.temporal_indices[idx // self.P]

        data = TensorDict({}, batch_size=[]) 
        for modality_id, reader in self.readers.items():
            p_index = self.spatial_indices[modality_id][idx % self.P]

            data_mod = reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

            data.set((modality_id,), data_mod)

            input_keys = self.inputs.get(modality_id, [])
            if input_keys:
                input_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in input_keys]
                data[("inputs", modality_id)] = torch.cat(input_tensors, dim=-1)

            diagnostic_keys = self.diagnostics.get(modality_id, [])
            if diagnostic_keys:
                diagnostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in diagnostic_keys]
                data[("diagnostics", modality_id)] = torch.cat(diagnostic_tensors, dim=-1)
            
            prognostic_keys = self.prognostics.get(modality_id, [])
            if prognostic_keys:
                prognostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in prognostic_keys]
                data[("prognostics", modality_id)] = torch.cat(prognostic_tensors, dim=-1)

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