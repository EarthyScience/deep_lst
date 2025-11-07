import os
print(os.getcwd())

import sys
print(sys.path)
sys.path.append("/Net/Groups/BGI/people/mwessel/wedata")  # for wedata imports without installation

from pathlib import Path

import pandas as pd
import torch
import time
import multiprocessing
import argparse

from eo_statistics.datareader import wedataReader, weDataset
from eo_statistics.helpers import iterate_statistics, to_scalar, save_statistics
from misc.utils import open_config
from torch.utils.data import DataLoader

num_cpus = multiprocessing.cpu_count()  # should be 64
print(f"Detected {num_cpus} CPUs available")

parser = argparse.ArgumentParser(description="Compute statistics for LST data.")
parser.add_argument('--modality', type=str, required=False,
                    default="era5",
                    help='Path to the experiment parquet file.')
parser.add_argument('--experiment', type=str, required=True,
                    help='Path to the experiment parquet file.')
parser.add_argument('--variable', type=str, required=True,
                    help='Variable to compute statistics for.')
parser.add_argument('--time_fraction', type=float, required=True,
                    help='Fraction of temporal samples to use for statistics.')
parser.add_argument('--space_fraction', type=float, required=True,
                    help='Fraction of spatial samples to use for statistics.')
parser.add_argument('--stratification', type=str, required=True,
                    help='Stratification variable for statistics.')
args = parser.parse_args()

config = open_config(experiment=args.experiment, path="./experiments")

eo_path = config['eo_path']

experiment_path = config['experiment_path'] + config[args.modality]['spatial_indices']
output_path = config['experiment_path'] + config[args.modality]['statistics']

print("Experiment path: ", experiment_path)
print("EO path:", eo_path)
print("Output path:", output_path)

reader = wedataReader(group = args.modality,
                    channel=args.variable,
                    config=config)

spatial_indices = reader.get_spatial_indices(frac=args.space_fraction)
temporal_indices = reader.get_temporal_indices(frac=args.time_fraction, stratify=args.stratification)

print("Temporal indices", temporal_indices)

dataset = weDataset(zarr_array=reader.channel,
                       spatial_indices=spatial_indices, 
                       temporal_indices=temporal_indices)

loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=32, 
                    persistent_workers=True, prefetch_factor=1, 
                    drop_last=False)


start_time = time.time()
statistics = iterate_statistics(loader, device='cpu')
end_time = time.time()
print(f"Computed statistics in {(end_time - start_time)/60:.2f} minutes.")
print(f"Statistics: {statistics}")
print(f"Mean: {to_scalar(statistics[0])}, Std: {to_scalar(statistics[1])}")
print("Stratification params: ", args.stratification, args.time_fraction, args.space_fraction)


save_statistics(statistics,
                variable_name=args.variable,
                stratification=args.stratification,
                time_fraction=args.time_fraction,
                space_fraction=args.space_fraction,
                output_path=output_path)

print(f"Saved statistics to {output_path}.")