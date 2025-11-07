import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from misc.filter import SpatialFilter, TemporalFilter

class wedataReader:
    def __init__(self, 
                 group: str,
                 channel: str,
                 config: dict,
                 lw_mask = True):

        self.group = group
        self.store = xr.open_zarr(config['eo_path'], group=self.group)
        self.config = config

        self.channel = self._select_channel(channel)

        experiment = pd.read_parquet(self.config["experiment_path"] + self.config[group]["spatial_indices"])
        self.spatial_indices = experiment[['row_idx', 'col_idx', 'lat_idx', 'lon_idx']].astype('int64')

        self.time_index = self._time_index()
        self.start_date = config['stats_start']
        self.end_date = config['stats_end']

        self.lw_mask = lw_mask

        self.spatial_filter = SpatialFilter(self.store, spatial_indices=self.spatial_indices)
        self.temporal_filter = TemporalFilter(self.store, time_index=self.time_index)

        print(f"wedataReader initialized for channel {channel} from {self.start_date} to {self.end_date}.")
        print(f"Channel dimensions: {self.channel.dims}")
        print(f"Channel shape: {self.channel.shape}")

    def _select_channel(self, channel: str): 
        channel_arr = self.store[channel]
        return channel_arr
    
    def _time_index(self):
        time_index = pd.DatetimeIndex(pd.date_range(start=self.config['index_start'], end=self.config['index_end'], freq='h'))
        return time_index

    def get_temporal_indices(self, frac=None, stratify = None):

        """
        Docstring for get_temporal_indices
        
        :param self: Description
        :param frac: Description
        :param stratify: Description
        """
        if "time" not in self.channel.dims:
            print("Data has no time dimension, skipping temporal indices.")
            return None
        
        self.temporal_indices = self.temporal_filter.apply_mask(self.start_date, self.end_date)
        print(self.temporal_indices)
        
        if frac is not None:
            if stratify is not None:
                print("Applying stratification for temporal indices.")
                self.temporal_indices = self.temporal_filter.apply_stratification(frac=frac, by=stratify)
            else:
                self.temporal_indices = self.temporal_filter.apply_rgn(frac=frac)

        return self.temporal_indices

    def get_spatial_indices(self, frac = None):
        
        
        if self.lw_mask:
            self.spatial_indices = self.spatial_filter.apply_mask(masking="land")
        if frac is not None:
            self.spatial_indices = self.spatial_filter.apply_rgn(frac)

        return self.spatial_indices

        

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

        if self.temporal_indices is not None:
            assert "time" in getattr(self.data, "dims", ()), \
                "temporal_indices given but data has no 'time' dimension"

        self.P = len(self.spatial_indices)
        self.T = len(self.temporal_indices) if self.temporal_indices is not None else 1
        self.samples = self.P * self.T

        print(f"Initialized StatsDataset with {self.samples} samples.")
        print(f"Initialized self.temporal_indices {self.temporal_indices}.")

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
            