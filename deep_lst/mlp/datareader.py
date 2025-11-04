import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class wedataReader:
    def __init__(self, 
                 config,
                 group,
                 eo_path: str,
                 lw_mask = True,
                 normalised=True):

        self.config = config
        self.group = group

        self.store = xr.open_zarr(eo_path, group=self.group)
        self.channels_source = self.config[self.group]["channels"]["source"]
        self.channels_target = self.config[self.group]["channels"]["target"]

        spatial_indices = pd.read_parquet(self.config[self.group]['spatial_indices'])
        self.spatial_indices = spatial_indices[['row_idx', 'col_idx', 'lat_idx', 'lon_idx']].astype('int64').reset_index(drop=True)

        self.temporal_indices = self._time_indices()

        self.statistics = pd.read_parquet(self.config[self.group]['statistics'])
        self._create_statistics_lookup()

        if lw_mask:
            self.apply_filtering()

        self.normalised = normalised

    def _create_statistics_lookup(self):
        self.mean_lookup = self.statistics.set_index('variable')["mean"]
        self.std_lookup = self.statistics.set_index('variable')["std"]
    
    def _time_indices(self):
        
        time_index = pd.date_range(start='2017-01-01 00:00', end='2022-12-31 23:00', freq='h')

        mask = (time_index >= self.config['train_start']) & (time_index < self.config['train_end'])
        time_idxs = np.nonzero(mask)[0].astype(np.int64)

        return time_idxs
    
    def get_indices(self, t_index, p_index):

        row_idx, col_idx, lat_idx, lon_idx = self.spatial_indices.iloc[p_index]
        time_idx = self.temporal_indices[t_index]
        return time_idx, lat_idx, lon_idx

    def get_data(self, sample_indices):

        X_dyn = self.store[self.channels_source["dynamic"]][sample_indices[0], sample_indices[1], sample_indices[2]]
        X_stat = self.store[self.channels_source["static"]][sample_indices[1], sample_indices[2]]
        X = X_dyn | X_stat

        y = self.store[self.channels_target["target"]][sample_indices[0], sample_indices[1], sample_indices[2]]

        if self.normalised:
            X = {self.normalise(X, var) for var in X.keys()}
            y = {self.normalise(y, var) for var in y.keys()}

        return X, y
    
    def normalise(self, data, var):
        return (data[var]-self.mean_lookup[var])/self.std_lookup[var]

    def denormalise(self, data, var):
        return data[var]*self.std_lookup[var] + self.mean_lookup[var]
    
    def apply_filtering(self):

        if self.group == "seviri":
            mask = "LWMASK"
        else:
            mask = "lsm"

        mask_arr = self.store[mask].values  # shape (lat, lon)

        self.land_mask = mask_arr[self.spatial_indices["lat_idx"].to_numpy(), self.spatial_indices["lon_idx"].to_numpy()] == 1
        df_land = self.spatial_indices.loc[self.land_mask].reset_index(drop=True)

        print(f"Retained {len(df_land)} / {len(self.spatial_indices)} points over land.")
        self.spatial_indices = df_land


class weDataset(Dataset):
    def __init__(self, 
                 config,
                 eo_path: str):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """

        self.config = config
        self.reader_era5 = wedataReader('era5', self.config, eo_path=eo_path)
        self.reader_seviri = wedataReader('seviri', self.config, eo_path=eo_path)

        self.P = len(self.reader_seviri.spatial_indices)
        self.T = len(self.reader_seviri.temporal_indices) 
        self.samples = self.P * self.T

        print(f"Initialized StatsDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = idx // self.P
        p_index = idx % self.P

        time_idx, lat_idx, lon_idx = self.reader_seviri.get_indices(t_index=t_index, p_index=p_index)
        X_seviri, target = self.reader_seviri.get_data((time_idx, lat_idx, lon_idx))

        time_idx, lat_idx, lon_idx = self.reader_era5.get_indices(t_index=t_index, p_index=p_index)
        X_era5, _ = self.reader_era5.get_data((time_idx, lat_idx, lon_idx))

        X_all = X_seviri | X_era5

        return X_all, target
    

