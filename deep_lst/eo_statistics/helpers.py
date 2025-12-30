import xarray as xr
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

import torch
from pathlib import Path

def load_statistics_npz(path: Path) -> dict[str, dict[str, float]]:
    path = Path(path)
    out: dict[str, dict[str, float]] = {}

    with np.load(path, allow_pickle=True) as z:
        variables = list(z["variables"])
        stat_names = [k for k in z.files if k != "variables"]

        for i, var in enumerate(variables):
            out[str(var)] = {}
            for stat in stat_names:
                out[str(var)][stat] = np.asarray(z[stat][i]).item()

    return out

    
def welford_incremental(prev_mean, prev_M2, prev_count, x_batch, float_dtype=torch.float32):
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
    if not torch.is_floating_point(x_batch):
        x_batch = x_batch.to(dtype=float_dtype)
    else:
        # Optionally standardize float precision
        x_batch = x_batch.to(dtype=float_dtype)

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
    nan_total = 0
    sample_total = 0

    counter = 0
    for b in data_loader:

        b_flat = b.view(-1)

        nan_total += torch.isnan(b_flat).sum().item()
        sample_total += b_flat.numel()

        b = b[~torch.isnan(b)]

        prev_mean, prev_M2, prev_count = welford_incremental(prev_mean, prev_M2, prev_count, b)
        min_total = min(min_total, b.min().item())
        max_total = max(max_total, b.max().item())

        counter += 1
        if counter % 500 == 0:
            print(f"Processing batch {counter}")

    mean, std = finalize_welford(prev_mean, prev_M2, prev_count)
    nan_ratio = nan_total / sample_total if sample_total > 0 else 0

    print("\n===== Overall Statistics =====")
    print(f"Total samples: {sample_total}")
    print(f"Total NaNs: {nan_total} ({nan_ratio:.2%})")
    print(f"Min: {min_total}")
    print(f"Max: {max_total}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")

    return mean, std, min_total, max_total, nan_total, sample_total


def to_scalar(x):
    """Convert tensors/arrays to Python floats/ints."""
    if hasattr(x, "item"):  # torch.Tensor or np scalar
        return x.item()
    if isinstance(x, (np.ndarray, list)) and np.size(x) == 1:
        return float(np.array(x).ravel()[0])
    return float(x) if np.isscalar(x) else np.nan

def save_statistics(statistics, variable_name, stratification, time_fraction, space_fraction, output_path):
    """
    Append a single variable's statistics to a Parquet file.
    """
    # Convert to DataFrame
    df = pd.DataFrame([{
        "variable": variable_name,
        "mean": to_scalar(statistics[0]),
        "std": to_scalar(statistics[1]),
        "min": to_scalar(statistics[2]),
        "max": to_scalar(statistics[3]),
        "nan_count": int(to_scalar(statistics[4])),
        "sample_count": int(to_scalar(statistics[5])),
        "stratification": stratification,
        "time_fraction": time_fraction,
        "space_fraction": space_fraction
    }])

    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append or create
    if output_path.exists():
        old = pd.read_parquet(output_path)
        df = pd.concat([old, df], ignore_index=True)

    df.to_parquet(output_path, index=False)
    print(f"Appended stats for {variable_name} to {output_path}")
