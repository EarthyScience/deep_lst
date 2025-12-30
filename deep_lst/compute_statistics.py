import os

from tensordict import TensorDict
print(os.getcwd())

import argparse

from wedata.eo_zarr import EOZarrStore
from wedata.statistics import StandardStatisticsMetric
from wedata.schemes.torch.base import VariableType

from eo_statistics.dataset import UnimodalDataset

import torch
from tqdm import tqdm
from torch.utils.data import DistributedSampler
import numpy as np
from pathlib import Path
import pandas as pd

def deep_to(obj, device, non_blocking=False):
    """
    Recursively move tensors in a nested structure to a device.
    Supports TensorDicts.
    """
    if isinstance(obj, TensorDict):
        return obj.apply(
            lambda x: x.to(device, non_blocking=non_blocking)
            if torch.is_tensor(x) else x,
            inplace=False,
        )
    return obj

def calculate_statistics(store: EOZarrStore, 
                        spatial_indices: np.ndarray,
                        device: str | torch.device = 'cpu',
                        **kwargs
                        ):
    
    reader = Reader(eozarr_store=store)
    reader.available_indices(indices=spatial_indices)

    device = torch.device(device)
    dataset = UnimodalDataset(
        spatial_indices=spatial_indices,
        reader=reader)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=0, rank=0, num_replicas=1)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs.get('batch_size', 32),
            sampler=sampler,
            num_workers=kwargs.get('num_workers', 1),
            drop_last=False,
            collate_fn=torch.stack
        )

    schema = reader.get_torch_schema()

    metrics = {}
    for name, var in schema.variables.items():
        if var.var_type == VariableType.CONTINUOUS:
            dim = (0,) + tuple(idx + 1 for idx in var.reduced_dimensions_indices)
            print("Dim when batch is added:", dim )
            metrics[name] = StandardStatisticsMetric(dim=dim).to(device=device)

    use_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for batch in tqdm(dataloader, disable=not use_tqdm, smoothing=0.005):
        batch = deep_to(batch, device = device, non_blocking=True)
        for key in batch.keys():
            print("Updating metric for variable:", key)
            print("Batch shape:", batch[key].shape)
            if metric := metrics.get(key):
                metric.update(batch[key])

    all_variables = {}
    for var_name, metric in metrics.items():
        stats = metric.compute()
        all_variables[var_name] = {k: v.cpu().numpy() for k, v in stats.items()}

    return all_variables

def save_statistics(statistics: dict, path: Path):
    rows = []
    for var, stats in statistics.items():
        row = {"variable": var}
        for k, v in stats.items():
            a = np.asarray(v)
            if a.ndim == 0:                 # scalar
                row[k] = a.item()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variable")
    df.to_parquet(path, engine="pyarrow")

def save_statistics_npz(statistics: dict, path: Path) -> None:
    """
    statistics: {var: {stat_name: scalar}}
    Writes one .npz with:
      - variables: array[str] of variable names (index)
      - each stat_name: array[float] aligned with `variables`
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    variables = np.array(list(statistics.keys()), dtype=object)

    # Collect all stat names (assumes scalars only)
    stat_names = sorted({k for stats in statistics.values() for k in stats.keys()})

    arrays: dict[str, np.ndarray] = {"variables": variables}

    for stat in stat_names:
        vals = []
        for var in variables:
            if stat not in statistics[var]:
                raise KeyError(f"Missing statistic '{stat}' for variable '{var}'.")
            vals.append(np.asarray(statistics[var][stat]).item())  # ensure scalar python value
        arrays[stat] = np.asarray(vals)

    np.savez(path, **arrays)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculate statistics for EOZarr dataset')
    parser.add_argument('-s', '--store', type=str, required=True, help='Path to the EOZarr store')
    parser.add_argument('-i', '--indices', type=str, required=True, help='Path to the sample indices file')
    parser.add_argument('-m', '--modality', type=str, required=True, help='Modality ID to calculate statistics for')
    parser.add_argument('--statistics', type=str, default='default', help='Path for saving the statistics')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for computation (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker threads for data loading')

    args = parser.parse_args()

    if args.modality== "era5":
        from wedata.data_reader.era5 import Era5Reader as Reader
    elif args.modality == "seviri":
        from wedata.data_reader.seviri import SeviriReader as Reader
    else:
        raise ValueError(f"Modality {args.modality} not supported.")

    store = EOZarrStore(args.store, mode='r')
    indices = np.load(args.indices)['mapping']

    stats = calculate_statistics(store, 
                                 indices, 
                                 device=args.device,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    
    for var_name, var_stats in stats.items():
        print(f"Variable: {var_name}")
        for stat_name, value in var_stats.items():
            print(f"  {stat_name}: {value}")
    
    save_statistics_npz(stats, Path(args.statistics))

    print(f"Saved statistics to {args.statistics}")