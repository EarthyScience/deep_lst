import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import time
import code

from pathlib import Path
from deep_lst.utils.time_filter import get_temporal_indices

from wedata.grids import T1H
from wedata.util.misc import assemble_statistics_from_npz
from wedata.eo_zarr import EOZarrStore
import wedata

from tensordict import TensorDict
from tensordict import stack as td_stack

def get_sample_indices(config: dict, split: str):

    spatial_idx = np.load(config["samples_path"])["mapping"]
    temporal_idx = np.arange(start=T1H.date_to_index(config[f'{split}_start']), stop=T1H.date_to_index(config[f'{split}_end']))

    space, time = np.meshgrid(spatial_idx, temporal_idx, indexing='ij')
    indices = np.stack([space.ravel(), time.ravel()], axis=1)
    print(indices.shape)
    return indices

def get_reader_from_modality(modality_id, config: dict, split: str):

    # Load assets
    statistics = assemble_statistics_from_npz(config["statistics_path"])
    indices = get_sample_indices(config, split) # or your way to load indices

    # Opens store
    store = EOZarrStore(config["store_path"], mode="r+")

    # Instantiate reader
    reader = wedata.get_data_reader(modality_id, eozarr_store=store, **config)
    reader.available_indices(indices=indices)
    reader.set_normalizer(statistics=statistics)

    return reader, indices


class UnimodalDataset(Dataset):
    def __init__(self, reader, indices, config):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.indices = indices
        self.reader = reader
        
        #self.P = len(self.spatial_indices)
        #self.T = len(self.temporal_indices)
        #self.samples = self.P * self.T

        print(f"Initialized Unimodal dataset with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        #t_index = self.temporal_indices[idx // self.P]
        #p_index = self.spatial_indices[idx % self.P]

        p_index, t_index = self.indices[idx]
        sample = self.reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

        return sample
    
    @classmethod
    def from_modality(cls,
        modality_id: str,
        config: dict,
        split: str,
    ):
        reader, indices = get_reader_from_modality(modality_id, config, split)

        dataset = cls(
            reader=reader,
            indices=indices,
            config=config
        )

        return dataset

class MultimodalDataset(Dataset):

    def __init__(self, reader: dict[str], indices: dict[str],  config):

        self.readers = reader
        self.indices = indices
        self.modality_ids = list(self.readers.keys())

        self.inputs = {mod : config[mod]['inputs'] for mod in self.modality_ids}
        self.diagnostics = {mod : config[mod]['diagnostics'] for mod in self.modality_ids}
        self.prognostics = {mod : config[mod]['prognostics'] for mod in self.modality_ids}

        print(f"Initialized MultimodalDataset with {len(self.indices[self.modality_ids[0]])} samples.")

    def __len__(self):
        return len(self.indices[self.modality_ids[0]])

    def __getitem__(self, idx):

        data = TensorDict({}, batch_size=[]) 

        for modality, reader in self.readers.items():

            p_index, t_index = self.indices[modality][idx]
            data_mod = reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

            data.set((modality,), data_mod)

            input_keys = self.inputs.get(modality, [])
            if input_keys:
                input_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in input_keys]
                data[("inputs", modality)] = torch.cat(input_tensors, dim=-1)

            diagnostic_keys = self.diagnostics.get(modality, [])
            if diagnostic_keys:
                diagnostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in diagnostic_keys]
                data[("diagnostics", modality)] = torch.cat(diagnostic_tensors, dim=-1)
            
            prognostic_keys = self.prognostics.get(modality, [])
            if prognostic_keys:
                prognostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in prognostic_keys]
                data[("prognostics", modality)] = torch.cat(prognostic_tensors, dim=-1)

        return data

    @classmethod
    def from_modalities(cls,
        #modality_ids: list[str],
        config: dict,
        split: str
    ):
        readers = {}
        indices = {}

        for modality, mod_config in config.items():
            reader, idxs_mod = get_reader_from_modality(modality, mod_config, split)
            readers[modality] = reader
            indices[modality] = idxs_mod

        dataset = cls(
            reader=readers,
            indices=indices,
            config=config,
        )

        return dataset