import os

from tensordict import TensorDict
print(os.getcwd())

import argparse

from wedata.eo_zarr import EOZarrStore
from wedata.statistics import StandardStatisticsMetric
from wedata.schemes.torch.base import VariableType

from deep_lst.dataset.dataset import UnimodalDataset
from deep_lst.utils.time_filter import get_temporal_indices
from deep_lst.utils.misc import *

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
                        reader: any,
                        spatial_indices: np.ndarray,
                        temporal_indices: np.ndarray,
                        device: str | torch.device = 'cpu',
                        **kwargs
                        ):
    
    reader = reader(eozarr_store=store)
    reader.available_indices(indices=spatial_indices)

    device = torch.device(device)
    dataset = UnimodalDataset(
        reader=reader,
        spatial_indices=spatial_indices,
        temporal_indices=temporal_indices
        )

    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=0)
    else:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False, seed=0, rank=0, num_replicas=1)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=kwargs['batch_size'],
            sampler=sampler,
            num_workers=kwargs['num_workers'],
            drop_last=False,
            collate_fn=torch.stack
        )

    schema = reader.get_torch_schema()

    metrics = {}
    for name, var in schema.variables.items():
        if var.var_type == VariableType.CONTINUOUS:
            dim = (0,) + tuple(idx + 1 for idx in var.reduced_dimensions_indices)
            metrics[name] = StandardStatisticsMetric(dim=dim).to(device=device)

    use_tqdm = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    for batch in tqdm(dataloader, disable=not use_tqdm, smoothing=0.005):
        batch = deep_to(batch, device = device, non_blocking=True)
        for key in batch.keys():
            if metric := metrics.get(key):
                metric.update(batch[key])

    all_variables = {}
    for var_name, metric in metrics.items():
        stats = metric.compute()
        all_variables[var_name] = {k: v.cpu().numpy() for k, v in stats.items()}

    return all_variables

def main():

    if 'RANK' in os.environ and 'LOCAL_RANK' in os.environ:
        torch.distributed.init_process_group(device_id=int(os.environ['LOCAL_RANK']))

    parser = argparse.ArgumentParser(description='Calculate statistics for EOZarr dataset')
    parser.add_argument('-s', '--store', type=str, required=True, help='Path to the EOZarr store')
    parser.add_argument('-i', '--indices', type=str, required=True, help='Path to the sample indices file')
    parser.add_argument('-m', '--modality', type=str, required=True, help='Modality ID to calculate statistics for')
    parser.add_argument('--statistics', type=str, default='default', help='Path for saving the statistics')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of worker threads for data loading')

    args = parser.parse_args()

    if torch.cuda.is_available() and torch.distributed.is_initialized():
        device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.modality== "era5":
        from wedata.data_reader.era5 import Era5Reader as Reader
    elif args.modality == "seviri":
        from wedata.data_reader.seviri import SeviriReader as Reader
    else:
        raise ValueError(f"Modality {args.modality} not supported.")

    CONFIG = open_config(experiment="test", path="experiments")

    store = EOZarrStore(args.store, mode='r')
    spatial_indices = np.load(args.indices)['mapping']

    temporal_indices = get_temporal_indices(CONFIG, split = 'train', stratify=False)

    stats = calculate_statistics(store, 
                                reader=Reader,
                                spatial_indices=spatial_indices, 
                                temporal_indices=temporal_indices,
                                device=device,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        for var_name, var_stats in stats.items():
            print(f"Variable: {var_name}")
            for stat_name, value in var_stats.items():
                print(f"  {stat_name}: {value}")
    
        save_statistics_npz(stats, Path(args.statistics))
        print(f"Saved statistics to {args.statistics}")


if __name__ == "__main__":

    main()