import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class wedataReader:
    def __init__(self, 
                 group: str,
                 channel: str,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 experiment_path: str,
                 eo_path: str,
                 lw_mask = True):

        self.group = group
        self.store = xr.open_zarr(eo_path, group=self.group)
        experiment = pd.read_parquet(experiment_path)
        self.source = experiment[['row_idx', 'col_idx', 'lat_idx', 'lon_idx']].astype('int64')
        self.start_date = start_date
        self.end_date = end_date
        self.channel = self._select_channel(channel)

        if lw_mask:
            self.apply_filtering()
            
        print(f"wedataReader initialized for channel {channel} from {start_date} to {end_date}.")
        print(f"Channel dimensions: {self.channel.dims}")
        print(f"Channel shape: {self.channel.shape}")

    def _select_channel(self, channel: str): 
        channel_arr = self.store[channel]
        return channel_arr
    
    def apply_filtering(self):

        if self.group == "seviri":
            mask = "LWMASK"
        else:
            mask = "lsm"

        mask_da = self.store[mask]
        print(f"Applying land water mask with dims: {mask_da.dims}, shape: {mask_da.shape}")

        # Drop time if present
        if "time" in mask_da.dims:
            print("Mask has a time dimension; using first timestep.")
            mask_da = mask_da.isel(time=0)

        mask_arr = mask_da.values  # shape (lat, lon)

        n_lat, n_lon = mask_arr.shape
        assert self.source["lat_idx"].max() < n_lat and self.source["lon_idx"].max() < n_lon, \
            f"indices out of range: lat_idx max={self.source['lat_idx'].max()}, lon_idx max={self.source['lon_idx'].max()}"


        land_mask = mask_arr[self.source["lat_idx"].to_numpy(), self.source["lon_idx"].to_numpy()] == 1
        df_land = self.source.loc[land_mask].reset_index(drop=True)

        print(f"Retained {len(df_land)} / {len(self.source)} points over land.")
        self.source = df_land

    def get_temporal_indices(self, frac):
        

        if "time" not in self.channel.dims:
            return None
        
        TIME_INDEX = pd.DatetimeIndex(self.channel['time'].values)

        # Half-open selection: [start, end)
        mask = (TIME_INDEX >= self.start_date) & (TIME_INDEX < self.end_date)
        self.time_idxs = np.nonzero(mask)[0].astype(np.int64)

        rng = np.random.default_rng(42)
        k = max(1, int(round(len(self.time_idxs) * frac))) # ensures at least one sample

        return np.sort(rng.choice(self.time_idxs, size=k, replace=False))


    def get_spatial_indices(self, frac):
        return self.source.sample(frac=frac, random_state=42).reset_index(drop=True)


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
            