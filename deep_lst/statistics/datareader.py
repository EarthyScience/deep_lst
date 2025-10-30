import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

class wedataReader:
    def __init__(self, 
                 group: str,
                 channel: str,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 experiment_path: str,
                 eo_path: str):

        self.store = xr.open_zarr(eo_path, group=group)
        experiment = pd.read_parquet(experiment_path)
        self.source = experiment[['row_idx', 'col_idx', 'lat_idx', 'lon_idx']].astype('int64')
        self.start_date = start_date
        self.end_date = end_date
        self.channel = self._select_channel(channel)

        print(f"wedataReader initialized for channel {channel} from {start_date} to {end_date}.")
        print(f"Channel dimensions: {self.channel.dims}")
        print(f"Channel shape: {self.channel.shape}")

    def _select_channel(self, channel: str): 
        channel_arr = self.store[channel]
        return channel_arr
    
    def apply_filtering(self, mask = "LWMASK"):

        mask_da = self.store[mask]
        print(f"Mask dims: {mask_da.dims}, shape: {mask_da.shape}")

        # Drop time if present
        if "time" in mask_da.dims:
            print("Mask has a time dimension; using first timestep.")
            mask_da = mask_da.isel(time=0)

        mask_arr = mask_da.values  # shape (lat, lon)
        n_lat, n_lon = mask_arr.shape

        assert self.source["lat_idx"].max() < n_lat and self.source["lon_idx"].max() < n_lon, \
            f"indices out of range: lat_idx max={self.source['lat_idx'].max()}, lon_idx max={self.source['lon_idx'].max()}"

        # Probe mask statistics
        unique_vals = np.unique(mask_arr)
        print(f"Unique mask values: {unique_vals[:20]}")

        land_mask = mask_arr[self.source["lat_idx"].to_numpy(), self.source["lon_idx"].to_numpy()] == 1

        df_land = self.source.loc[land_mask].reset_index(drop=True)

        print(f"Retained {len(df_land)} / {len(self.source)} points over land.")
        self.source = df_land

    def get_temporal_indices(self):
        
        if 'time' in self.channel.dims:
            TIME_INDEX = pd.DatetimeIndex(self.channel['time'].values)

            # Half-open selection: [start, end)
            mask = (TIME_INDEX >= self.start_date) & (TIME_INDEX < self.end_date)
            self.time_idxs = np.nonzero(mask)[0].astype(np.int64)
            return self.time_idxs
        else:
            return None

    def get_spatial_indices(self):
        return self.source.reset_index(drop=True)
