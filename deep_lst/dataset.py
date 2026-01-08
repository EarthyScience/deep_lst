import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

from pathlib import Path
from deep_lst.utils.time_filter import get_temporal_indices
from wedata.eo_zarr import EOZarrStore

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
        self.modalities = list(self.readers.keys())
        self.config = config

        self.P = len(self.spatial_indices[self.modalities[0]])
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        print(f"Initialized MultimodalDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        data = {}
        for modality_id, reader in self.readers.items():
            t_index = self.temporal_indices[idx // self.P]
            p_index = self.spatial_indices[modality_id][idx % self.P]

            data[modality_id] = reader.get_data_single(space_idx=p_index, time_idx=t_index)
        
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