import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from deep_lst.misc.filter import SpatialFilter, TemporalFilter
from deep_lst.misc.utils import build_feature_list

class wedataReader:
    def __init__(self, 
                 config,
                 group,
                 landmask = True,
                 normalised: bool = True,
                 split: str = "train"):

        self.config = config
        self.group = group
        self.split = split
        
        self.store = xr.open_zarr(self.config["eo_path"], group=self.group)
        self.channels_input = self.config[self.group]["channels"]["input"]
        self.channels_target = self.config[self.group]["channels"]["target"]
        self.use_input = self.config[self.group]['use_input']

        spatial_indices = pd.read_parquet(self.config["experiment_path"] + self.config[group]["spatial_indices"])
        self.spatial_indices = spatial_indices[['row_idx', 'col_idx', 'lat_idx', 'lon_idx']].astype('int64').reset_index(drop=True)

        self.temporal_indices = self._time_indices()

        self.statistics = pd.read_parquet(self.config["experiment_path"] + self.config[group]['statistics'])
        self._create_statistics_lookup()

        self.normalised = normalised

        if landmask:
            self._create_landmask()

    def _create_statistics_lookup(self):
        self.mean_lookup = self.statistics.set_index('variable')["mean"]
        self.std_lookup = self.statistics.set_index('variable')["std"]
    
    def _time_indices(self):

        time_index = pd.DatetimeIndex(pd.date_range(start=self.config['index_start'], end=self.config['index_end'], freq='h'))

        mask = (time_index >= self.config[f"{self.split}_start"]) & (time_index < self.config[f"{self.split}_end"])
        time_idxs = np.nonzero(mask)[0].astype(np.int64)

        return time_idxs
    
    def get_indices(self, t_index, p_index):

        row_idx, col_idx, lat_idx, lon_idx = self.spatial_indices.iloc[p_index]
        time_idx = self.temporal_indices[t_index]
        return (time_idx, lat_idx, lon_idx)
    
    def get_input(self, sample_indices):
        """
        Returns inputs according to self.use_input:
        - if 'dynamic' in use_input: include dynamic vars
        - if 'static' in use_input: include static vars
        - if both are present: merged dict of both
        """
        
        selected = self.use_input if isinstance(self.use_input, (list, tuple, set)) else [self.use_input]

        data = {}
        if 'dynamic' in selected:
            data.update(self._get_data(sample_indices, type='dynamic'))
        if 'static' in selected:
            data.update(self._get_data(sample_indices, type='static'))
        return data

    def _get_data(self, sample_indices, type):
        if type == "dynamic":
            data = {var: torch.from_numpy(self.store[var][sample_indices[0], sample_indices[1], sample_indices[2]].values) for var in self.channels_input[type]}
        else:
            data = {var: torch.from_numpy(self.store[var][sample_indices[1], sample_indices[2]].values) for var in self.channels_input[type]}
        if self.normalised:
            data = {var: self.normalise(data, var) for var in data.keys()}
        return data

    def get_target(self, sample_indices, type= None):
        data = {var: torch.from_numpy(self.store[var][sample_indices[0], sample_indices[1], sample_indices[2]].values) for var in self.channels_target}
        if self.normalised:
            data = {var: self.normalise(data, var) for var in data.keys()}
        return data
    
    def normalise(self, data, var):
        return (data[var]-self.mean_lookup[var])/self.std_lookup[var]

    def denormalise(self, data, var):
        return data[var]*self.std_lookup[var] + self.mean_lookup[var]

    def _create_landmask(self):

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
                 split: str = "train"
                 ):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """

        self.config = config
        self.split = split

        self.reader_era5 = wedataReader(group='era5', 
                                        config=self.config, 
                                        split=self.split,
                                        normalised=True)
        self.reader_seviri = wedataReader(group='seviri', 
                                           config=self.config,
                                           split=self.split,
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

        self.feature_list = build_feature_list(self.config) 

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = idx // self.P
        p_index = idx % self.P

        sample_indices_seviri = self.reader_seviri.get_indices(t_index=t_index, p_index=p_index)
        input_seviri = self.reader_seviri.get_input(sample_indices_seviri)

        sample_indices_era5 = self.reader_era5.get_indices(t_index=t_index, p_index=p_index)
        input_era5 = self.reader_era5.get_input(sample_indices_era5)

        input_combined = input_seviri | input_era5

        target = self.reader_seviri.get_target(sample_indices_seviri)

        return input_combined, target
    
    def apply_landmask(self, spatial_indices: pd.DataFrame, landmask: pd.DataFrame, col='mask'):
        si = spatial_indices.merge(landmask, on=['row_idx','col_idx'], how='inner')
        si = si.loc[si[col]].drop(columns=[col]).reset_index(drop=True)
        return si
    

