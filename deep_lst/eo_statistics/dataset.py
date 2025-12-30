import pandas as pd
from torch.utils.data import Dataset

class UnimodalDataset(Dataset):
    def __init__(self, spatial_indices, reader, temporal_indices = None):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.spatial_indices = spatial_indices
        self.reader = reader

        # only for degbugging change in the future
        if temporal_indices is not None:
            self.temporal_indices = temporal_indices
        else:
            self.temporal_indices = _time_index(idx_start="2017-01-01 00:00", idx_end="2017-01-01 02:00")
        
        self.P = len(self.spatial_indices)
        self.T = len(self.temporal_indices)
        self.samples = self.P * self.T

        print(f"Initialized StatsDataset with {self.samples} samples.")
        print(f"Initialized self.temporal_indices {self.temporal_indices}.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):

        t_index = idx // self.P
        p_index = self.spatial_indices[idx % self.P]

        sample = self.reader.get_data_single(spatial_idx=p_index, time_idx=t_index)
        
        return sample


def _time_index(idx_start, idx_end, config=None):
    time_index = pd.DatetimeIndex(pd.date_range(start=idx_start, end=idx_end, freq='h'))
    return time_index