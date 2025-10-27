import torch
from torch.utils.data import Dataset

class weDataset(Dataset):
    def __init__(self, zarr_array, indices, source):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.data = zarr_array
        self.indices = indices
        self.source = source

        self.mapping = {(row, col): (lat, lon) for row, col, lat, lon in zip(
            self.source['row_idx'], self.source['col_idx'], self.source['lat_idx'], self.source['lon_idx']
            )}
        
        print(f"Initialized StatsDataset with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        t, row, col = self.indices.iloc[idx]
        lat_idx, lon_idx = self.mapping[(row, col)]
        x = self.data[t, lat_idx, lon_idx].values  # lazy-load element, era5 order time, lat, lon

        return x