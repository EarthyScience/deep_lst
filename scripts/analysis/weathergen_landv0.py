"""PARKED ANALYSIS SCRIPT (moved out of the package 2026-07-28).

One-off WeatherGenerator comparison; runs its whole pipeline at import time against
hardcoded paths. Its imports reference the deleted legacy metrics layer
(compute_pooled_by_day / pool_metric) -- port to the sufficient-statistics chain
(metrics.sufficient_stats / pool_stats / measures_from_stats) before reuse.
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from eodata.grids.majortom import MajorTomGrid, MajorTomLandGrid

STORE = "/Net/Groups/BGI/tscratch/weathergenerator/tmp/inference_lst/train_lst.zarr"
OUT = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/WGL/v1/weathergen_landv0.nc"

z = zarr.open(STORE, mode="r")
lst = z["lst"][:]  # (10825, 7, 24, 1)
mask = z["mask"][:]
ft = pd.to_datetime(np.char.decode(z["forecast_time"][:], "utf-8"))
sidx = z["spatial_index"][:]

print("lst.shape", lst.shape, "mask.shape", mask.shape, "ft.shape", ft.shape, "sidx.shape", sidx.shape)

n = lst.shape[0]
lst = lst.reshape(n, 7 * 24)  # 168 lead-hours
mask = mask.reshape(n, 7 * 24)
lead = pd.timedelta_range("0h", periods=7 * 24, freq="h")

land = MajorTomLandGrid(5)  # his idx = LOCAL land position
gidx = land._indices[sidx]
c = np.asarray(MajorTomGrid(5).centers)  # col0=lon, col1=lat
lat, lon = c[gidx, 1], c[gidx, 0]

# pivot flat (sample) -> (init_time, lead_time, pixel); NaN where (init, pixel) unsampled
init = pd.DatetimeIndex(np.unique(ft))
pix, first = np.unique(sidx, return_index=True)
ipos = init.get_indexer(ft)
ppos = np.searchsorted(pix, sidx)

lst_cube = np.full((len(init), 7 * 24, len(pix)), np.nan, dtype=np.float32)
mask_cube = np.zeros((len(init), 7 * 24, len(pix)), dtype=np.float32)
lst_cube[ipos, :, ppos] = lst
mask_cube[ipos, :, ppos] = mask

ds = xr.Dataset(
    {"lst": (("init_time", "lead_time", "pixel"), lst_cube), "mask": (("init_time", "lead_time", "pixel"), mask_cube)},
    coords={
        "init_time": init,
        "lead_time": lead,
        "lat": ("pixel", lat[first]),
        "lon": ("pixel", lon[first]),
        "global_idx": ("pixel", gidx[first]),
        "spatial_index": ("pixel", pix),
    },
)
print("forecast grid", dict(ds.sizes))

ds.to_netcdf(OUT)
print("saved", OUT)
