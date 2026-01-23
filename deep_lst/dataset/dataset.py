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
    def __init__(self, reader, indices):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.indices = indices
        self.reader = reader

        print(f"Initialized Unimodal dataset with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

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

    def __init__(self, reader: dict[str], indices: dict[str],  **kwargs):

        self.readers = reader
        self.indices = indices
        self.kwargs = kwargs
        self.samples = len(self.indices[next(iter(self.readers))])
        print(f"Initialized MultimodalDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        data = TensorDict({}, batch_size=[]) 

        for modality, reader in self.readers.items():

            p_index, t_index = self.indices[modality][idx]
            data_mod = reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

            # This will change with structured dtype where concatenation isnt needed anymore
            if self.kwargs[modality]['inputs']:
                input_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in self.kwargs[modality]['inputs']]
                data[("inputs", modality)] = torch.cat(input_tensors, dim=-1)

            if self.kwargs[modality]['diagnostics']:
                diagnostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in self.kwargs[modality]['diagnostics']]
                data[("diagnostics", modality)] = torch.cat(diagnostic_tensors, dim=-1)
            
            if self.kwargs[modality]['prognostics']:
                prognostic_tensors = [data_mod[k] if k in data_mod else data[k] 
                           for k in self.kwargs[modality]['prognostics']]
                data[("prognostics", modality)] = torch.cat(prognostic_tensors, dim=-1)

        return data

    @classmethod
    def from_modalities(cls,
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
            **config,
        )

        return dataset