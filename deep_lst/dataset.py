import pandas as pd
from torch.utils.data import Dataset
import torch

class UnimodalDataset(Dataset):
    def __init__(self, reader, spatial_indices, temporal_indices):
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

        print(f"Initialized StatsDataset with {self.samples} samples.")
        print(f"Initialized self.temporal_indices {self.temporal_indices}.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = self.temporal_indices[idx // self.P]
        p_index = self.spatial_indices[idx % self.P]

        sample = self.reader.get_data_single(spatial_idx=p_index, time_idx=t_index)
        
        #sample['indices'] = torch.from_numpy((t_index, p_index))

        return sample

class MultimodalDataset(Dataset):

    def __init__(self, reader: dict[str, UnimodalDataset], spatial_indices, temporal_indices):

        self.readers = reader
        self.modalities = list(readers.keys())
        self.length = min(len(reader) for reader in readers.values())
        print(f"Initialized MultimodalDataset with {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {}
        for modality_id, reader in self.readers.items():
            data[modality_id] = reader[idx]
        return data