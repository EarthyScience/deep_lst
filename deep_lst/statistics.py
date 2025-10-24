import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch

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
    """Return mean and standard deviation from aggregate."""
    if count < 2:
        return mean, torch.zeros_like(M2)
    sample_var = (M2 / (count - 1)).sqrt()  # unbiased sample variance
    return mean.numpy(), sample_var.numpy()

def iterate_statistics(data_loader, device='cpu'):
    """
    Iterate over data loader to compute mean and std using Welford's algorithm.
    
    Args:
        data_loader: PyTorch DataLoader yielding batches of data
        device: device to perform computations on ('cpu' or 'cuda')
    Returns:
        mean, std: computed mean and standard deviation arrays
    """

    prev_mean, prev_M2, prev_count = 0, 0, 0
    min_total = float('inf')
    max_total = float('-inf')

    counter = 0
    for b in data_loader:

        b = b[~torch.isnan(b)]
        print(f"Processed batch of size {b.shape}, after removing NaNs: {b.shape}")

        prev_mean, prev_M2, prev_count = welford_incremental(prev_mean, prev_M2, prev_count, b)
        min_total = min(min_total, b.min().item())
        max_total = max(max_total, b.max().item())

        counter += 1
        if counter % 500 == 0:
            print(f"Processing batch {counter}")

    mean, std = finalize_welford(prev_mean, prev_M2, prev_count)
    print("Overall min:", min_total)
    print("Overall max:", max_total)
    print("Overall mean:", mean)
    print("Overall std:", std)
    return mean, std, min_total, max_total