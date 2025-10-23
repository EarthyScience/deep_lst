import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

class ERA5reader:
    def __init__(self, 
                 channel: str ,
                 start_date: pd.Timestamp,
                 end_date: pd.Timestamp,
                 experiment_path: str,
                 eo_path: str):

        self.store = xr.open_zarr(eo_path, group="era5")
        self.source = pd.read_parquet(experiment_path)
        self.start_date = start_date
        self.end_date = end_date
        self.channel = self._select_channel(channel)

        print(f"ERA5reader initialized for channel {channel} from {start_date} to {end_date}.")
        print(f"Channel dimensions: {self.channel.dims}")
        print(f"Channel shape: {self.channel.shape}")

    def _select_channel(self, channel: str = "skt"):
        channel_arr = self.store[channel]
        return channel_arr

    def get_indices(self):
        
        TIME_INDEX = self.channel['time'].to_index() 
        self.time_idx = TIME_INDEX.slice_indexer(start=self.start_date, end=self.end_date)
        num_times = self.time_idx.stop - self.time_idx.start

        df_sel = self.source[['row_idx', 'col_idx']].copy()

        # Repeat each row/col combination for each time step
        # Tile the time indices for all points
        combined_indices = pd.DataFrame({
            'time_idx': np.tile(np.arange(self.time_idx.start, self.time_idx.stop), len(self.source)),
            'row_idx': np.repeat(df_sel['row_idx'].values, num_times),
            'col_idx': np.repeat(df_sel['col_idx'].values, num_times)
        })

        return combined_indices
