import torch
from torch.utils.data import Dataset

class weDataset(Dataset):
    def __init__(self, zarr_array, spatial_indices, temporal_indices = None):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.data = zarr_array
        self.spatial_indices = spatial_indices
        self.temporal_indices = temporal_indices

        
        self.P = len(self.spatial_indices)
        self.T = len(self.temporal_indices) if self.temporal_indices is not None else 1
        self.samples = self.P * self.T

        print(f"Initialized StatsDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = idx // self.P
        p_index = idx % self.P

        row, col, lat_idx, lon_idx = self.spatial_indices.iloc[p_index]

        if self.temporal_indices is not None:
            t = self.temporal_indices[t_index]
            x = self.data[t, lat_idx, lon_idx].values  # lazy-load element order time, lat, lon
            return x

        else:
            x = self.data[lat_idx, lon_idx].values 
            return x
            