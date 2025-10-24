import os
print(os.getcwd())

import sys
print(sys.path)
sys.path.append("/Net/Groups/BGI/people/mwessel/wedata")  # for wedata imports without installation

from wedata.typehints import StatisticsDict
from wedata.modalities import ModalityId
from wedata.eo_zarr import EOZarrStore

import pandas as pd
import torch
import time
import multiprocessing

from deep_lst.datasets import StatsDataset
from deep_lst.datareader import ERA5reader
from deep_lst.statistics import iterate_statistics
from torch.utils.data import DataLoader

num_cpus = multiprocessing.cpu_count()  # should be 64
print(f"Detected {num_cpus} CPUs available")

eo_path = "/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/test2.zarr"
experiment_path = "/Net/Groups/BGI/people/mwessel/deep_lst/experiments/filtered_seviri.parquet"

reader = ERA5reader(group = "seviri",
                    channel="LST",
                    start_date=pd.Timestamp('2019-01-01 00:00'),
                    end_date=pd.Timestamp("2019-01-01 12:00"),
                    experiment_path= experiment_path,
                    eo_path=eo_path)

dataset = StatsDataset(zarr_array=reader.channel,
                       indices=reader.get_indices(),
                       source=reader.source)

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
    "max": statistics[3]
}

#eo_store = EOZarrStore(eo_path)
#eo_store.set_statistics(ModalityId.ERA5, "LST", stats)