"""Climatology vs observed LST at the training sites over one test period.

Observed LST + quality flags are read from the full seviri store at the MTG training
anchors (the same nearest-neighbour mapping the training preload uses); the climatology
is the precomputed hourly baseline at the same sites (baselines/climatology_hourly).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H

from deep_lst.exploration.full_store import SEVIRI_STORE, load_store_columns

CLIM_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/climatology_hourly/eval_mtg.nc"
OUT_PATH = "/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/climatology.png"

PERIOD_START, PERIOD_END = "2024-11-04", "2024-11-17"  # test_period_22, inclusive

VALID_FLAGS = [101.0, 102.0, 301.0, 302.0]
INVALID_FLAGS = [103.0]

# --- climatology ---
clim = xr.open_dataset(CLIM_PATH)
clim_ts = clim["LST"].sel(valid_time=slice(PERIOD_START, PERIOD_END)).mean("pixel")

# --- observed ---
day_lo = T1H.date_to_index(PERIOD_START) // 24
day_hi = T1H.date_to_index(PERIOD_END) // 24 + 1  # exclusive
columns, _, inverse, _ = load_store_columns(
    "seviri",
    SEVIRI_STORE,
    "mtg",
    {"data": ["LST"], "metadata": ["quality_flag"]},
    anchor="mtg",
    day_range=(day_lo, day_hi),
    num_workers=16,
)
times = pd.date_range(PERIOD_START, periods=(day_hi - day_lo) * 24, freq="h")
# [inverse] expands the unique seviri cells back to one row per training anchor.
lst = xr.DataArray(columns["data"].numpy()[inverse, :, 0], dims=("pixel", "time"), coords={"time": times})
qf = xr.DataArray(columns["metadata"].numpy()[inverse, :, 0], dims=("pixel", "time"), coords={"time": times})

lst_valid = lst.where(qf.isin(VALID_FLAGS)).mean("pixel")
lst_invalid = lst.where(qf.isin(INVALID_FLAGS)).mean("pixel")

# --- plot ---
fig, ax = plt.subplots(figsize=(14, 4))

ax.plot(clim_ts["valid_time"].values, clim_ts.values, color="gray", linestyle="--", linewidth=1.2, label="Climatology")
ax.plot(lst_valid["time"].values, lst_valid.values, color="red", linewidth=1.0, label="Observed valid (QF 1–2)")
ax.plot(
    lst_invalid["time"].values,
    lst_invalid.values,
    color="lightblue",
    linewidth=1.0,
    label="Observed degraded (QF 3)",
)

ax.set_ylabel("LST (K)")
ax.set_title("LST climatology vs. observed — test_period_22 (2024-11-04 → 2024-11-17)")
ax.legend()
fig.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.show()


# positional (.values) difference: both series cover the same hours of the period
print(np.sqrt(np.nanmean((lst_valid.values - clim_ts.values) ** 2)))
print(np.nanmean(abs(lst_valid.values - clim_ts.values)))
