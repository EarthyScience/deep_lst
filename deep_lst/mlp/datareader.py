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
                 landmask = True,
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

        if landmask:
            self.set_landmask()

        self.normalised = normalised

    def _create_statistics_lookup(self):
        self.mean_lookup = self.statistics.set_index('variable')["mean"]
        self.std_lookup = self.statistics.set_index('variable')["std"]
    
    def _time_indices(self):
        
        time_index = pd.DatetimeIndex(pd.date_range(start='2017-01-01 00:00', end='2022-12-31 23:00', freq='h'))

        mask = (time_index >= self.config['train_start']) & (time_index < self.config['train_end'])
        time_idxs = np.nonzero(mask)[0].astype(np.int64)

        return time_idxs
    
    def get_indices(self, t_index, p_index):

        row_idx, col_idx, lat_idx, lon_idx = self.spatial_indices.iloc[p_index]
        time_idx = self.temporal_indices[t_index]
        return time_idx, lat_idx, lon_idx

    def get_data(self, sample_indices, type):

        if type == "source":
            X_dyn = {var: torch.from_numpy(self.store[var][sample_indices[0], sample_indices[1], sample_indices[2]].values) for var in self.channels_source["dynamic"]}
            X_stat = {var: torch.from_numpy(self.store[var][sample_indices[1], sample_indices[2]].values) for var in self.channels_source["static"]}
            data = X_dyn | X_stat

        if type == "target":
            data = {var: torch.from_numpy(self.store[var][sample_indices[0], sample_indices[1], sample_indices[2]].values) for var in self.channels_target}

        if self.normalised:
            data = {var: self.normalise(data, var) for var in data.keys()}

        return data
    
    def normalise(self, data, var):
        return (data[var]-self.mean_lookup[var])/self.std_lookup[var]

    def denormalise(self, data, var):
        return data[var]*self.std_lookup[var] + self.mean_lookup[var]
    
    def set_landmask(self):

        if self.group == "seviri":
            mask = "LWMASK"
        else:
            mask = "lsm"

        mask_arr = self.store[mask].values  # shape (lat, lon)

        land_mask = mask_arr[self.spatial_indices["lat_idx"].to_numpy(), self.spatial_indices["lon_idx"].to_numpy()] == 1

        self.land_mask = pd.DataFrame({
            'row_idx': self.spatial_indices['row_idx'].to_numpy(),
            'col_idx': self.spatial_indices['col_idx'].to_numpy(),
            'mask': land_mask
        })


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

        self.reader_era5 = wedataReader(group='era5', 
                                        config=self.config, 
                                        eo_path=eo_path,
                                        normalised=True)
        self.reader_seviri = wedataReader(group='seviri', 
                                           config=self.config, 
                                           eo_path=eo_path,
                                           normalised=True)

        self.reader_seviri.spatial_indices = self.apply_landmask(
            self.reader_seviri.spatial_indices,
            self.reader_seviri.land_mask,
            col='mask'
        )
        self.reader_era5.spatial_indices = self.apply_landmask(
            self.reader_era5.spatial_indices,
            self.reader_seviri.land_mask,
            col='mask'
        )

        self.P = len(self.reader_seviri.spatial_indices)
        self.T = len(self.reader_seviri.temporal_indices) 
        self.samples = self.P * self.T

        print(f"Initialized weDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = idx // self.P
        p_index = idx % self.P

        time_idx, lat_idx, lon_idx = self.reader_seviri.get_indices(t_index=t_index, p_index=p_index)
        source_seviri = self.reader_seviri.get_data((time_idx, lat_idx, lon_idx), type="source")
        target = self.reader_seviri.get_data((time_idx, lat_idx, lon_idx), type="target")

        time_idx, lat_idx, lon_idx = self.reader_era5.get_indices(t_index=t_index, p_index=p_index)
        source_era5 = self.reader_era5.get_data((time_idx, lat_idx, lon_idx), type="source")

        source_all = source_seviri | source_era5

        return source_all, target

    def apply_landmask(self, spatial_indices: pd.DataFrame, landmask: pd.DataFrame, col='mask'):
        si = spatial_indices.merge(landmask, on=['row_idx','col_idx'], how='inner')
        si = si.loc[si[col]].drop(columns=[col]).reset_index(drop=True)
        return si
    

