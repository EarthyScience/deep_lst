import argparse

import matplotlib.pyplot as plt
import xarray as xr

from deep_lst.utils import ExperimentPaths

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
parser.add_argument("--version", type=str, required=True, help="Experiment version")
args = parser.parse_args()

paths = ExperimentPaths()
ds = xr.open_dataset(paths.predictions_path(args.experiment, args.version))

ds.isel(init_time=0).plot.scatter(x="LST_obs", y="LST_pred", col="lead_time")
plt.savefig(paths.plots_path(args.experiment, args.version, "scatter"), dpi=150, bbox_inches="tight")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i in range(5):
    ds["LST_obs"].isel(init_time=0, lead_time=i).plot(ax=axes[0, i], vmin=-1, vmax=1)
    ds["LST_pred"].isel(init_time=0, lead_time=i).plot(ax=axes[1, i], vmin=-1, vmax=1)
axes[0, 0].set_ylabel("Obs")
axes[1, 0].set_ylabel("Pred")
plt.tight_layout()
plt.savefig(paths.plots_path(args.experiment, args.version, "maps"), dpi=150, bbox_inches="tight")
