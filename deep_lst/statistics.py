import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from code.datasets import StatsDataset
from code.datareader import ERA5reader

reader = ERA5reader(channel="skt",
                    start_date=pd.Timestamp('2019-01-01 00:00'),
                    end_date=pd.Timestamp("2019-01-03 23:00"))

dataset = StatsDataset(zarr_array=reader.channel,
                       indices=reader.get_indices(),
                       source=reader.source)

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)


start = time.time()
skt = reader.channel.isel(time=reader.time_idx) #, latitude = df['lat_idx'], longitude=df['lon_idx'])

mean_global = skt.mean(dim=('time', 'latitude', 'longitude'), skipna=True).compute()
std_global  = skt.std(dim=('time', 'latitude', 'longitude'), skipna=True).compute()

print(f"Computation took {(time.time() - start)/60:.2f} minutes")
print("Global mean", mean_global)
print("Global std", std_global)



def welford_incremental(prev_mean, prev_M2, prev_count, x_batch):
    """
    Vectorized incremental update of mean and M2 for a new batch.
    
    Args:
        prev_mean: previous mean tensor
        prev_M2: previous sum-of-squares tensor
        prev_count: previous sample count (scalar)
        x_batch: new batch tensor of shape [batch, ...]
    
    Returns:
        new_mean, new_M2, new_count
    """
    batch_count = x_batch.shape[0]
    batch_mean = x_batch.mean(dim=0)
    batch_delta = x_batch - batch_mean
    batch_M2 = (batch_delta**2).sum(dim=0)
    
    # Combined statistics
    total_count = prev_count + batch_count
    delta = batch_mean - prev_mean
    new_mean = prev_mean + delta * batch_count / total_count
    new_M2 = prev_M2 + batch_M2 + delta**2 * prev_count * batch_count / total_count
    
    return new_mean, new_M2, total_count


def finalize_welford(mean, M2, count):
    """Return mean, variance, and sample variance from aggregate."""
    if count < 2:
        return mean, torch.zeros_like(M2), torch.zeros_like(M2)
    var = M2 / count           # biased population variance
    sample_var = M2 / (count - 1)  # unbiased
    return mean, var, sample_var

