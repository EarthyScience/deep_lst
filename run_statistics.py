import os
print(os.getcwd())

import sys
print(sys.path)
sys.path.append("/Net/Groups/BGI/people/mwessel/wedata")  # for wedata imports without installation

from pathlib import Path
from wedata.typehints import StatisticsDict
from wedata.modalities import ModalityId
from wedata.eo_zarr import EOZarrStore

import pandas as pd
import torch
import time
import multiprocessing
import argparse

from deep_lst.statistics.datasets import weDataset
from deep_lst.statistics.datareader import wedataReader
from deep_lst.statistics.helpers import iterate_statistics, save_statistics
from torch.utils.data import DataLoader

num_cpus = multiprocessing.cpu_count()  # should be 64
print(f"Detected {num_cpus} CPUs available")

parser = argparse.ArgumentParser(description="Compute statistics for LST data.")
parser.add_argument('--modality', type=str, required=False,
                    default="era5",
                    help='Path to the experiment parquet file.')
parser.add_argument('--experiment', type=str, required=False,
                    default="",
                    help='Path to the experiment parquet file.')
parser.add_argument('--eo_path', type=str, required=False,
                    default="/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/seviri_lst.zarr",
                    help='Path to the EO Zarr store.')
parser.add_argument('--variable', type=str, required=True,
                    help='Variable to compute statistics for.')
args = parser.parse_args()

eo_path = args.eo_path



experiment_path = f"/Net/Groups/BGI/people/mwessel/deep_lst/experiments/new_{args.modality}{args.experiment}.parquet"
output_path = f"/Net/Groups/BGI/people/mwessel/deep_lst/experiments/new_{args.modality}{args.experiment}_statistics.parquet"

reader = wedataReader(group = args.modality,
                    channel=args.variable,
                    start_date=pd.Timestamp('2017-01-01 00:00'),
                    end_date=pd.Timestamp("2017-01-31 23:00"),
                    experiment_path= experiment_path,
                    eo_path=eo_path)

dataset = weDataset(zarr_array=reader.channel,
                       spatial_indices=reader.get_spatial_indices(), 
                       temporal_indices=reader.get_temporal_indices())

loader = DataLoader(dataset, batch_size=256*4, shuffle=True, num_workers=num_cpus-2, 
                    persistent_workers=True, prefetch_factor=1, 
                    drop_last=False)


start_time = time.time()
statistics = iterate_statistics(loader, device='cpu')
end_time = time.time()
print(f"Computed statistics in {(end_time - start_time)/60:.2f} minutes.")

stats: StatisticsDict = {
    "mean": statistics[0],
    "std": statistics[1],
    "min": statistics[2],
    "max": statistics[3],
    "nan_count": statistics[4],
    "sample_count": statistics[5]
}

save_statistics(stats, variable_name=args.variable, output_path=output_path)

#eo_store = EOZarrStore(eo_path)
#eo_store.set_statistics(ModalityId.ERA5, "LST", stats)