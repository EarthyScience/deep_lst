"""Yearly mean fractional vegetation cover on the seviri equiangular grid.

FVC already lives on the seviri grid in the full store, so there is no regridding
here: this is a temporal mean over one year, written out in the same layout as
aridity_index.py so both files can be indexed with the same spatial assets npz.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from eodata.grids import T1H

from deep_lst.exploration.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

OUT_DIR = Path("/Net/Groups/BGI/tscratch/weathergenerator/aux/AI_ET0__annual_v3_1")
OUT_PATH = OUT_DIR / "africa_fvc_2024_equiangular_005.nc"
PLOT_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/vegetation_cover")

INDICES_PATH = "assets/spatial/indices_sites_fvc.npz"
EVAL_REGIONS = ["eval_dahra", "eval_gobabeb", "eval_bouamir", "eval_yangambi", "eval_johannesburg"]

YEAR = 2024
ROW_BLOCK = 100_000  # rows read per pass; the whole year for all rows would be ~15 GB

PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- yearly mean over the store's daily FVC ----------------------------------

grid = store_grid("seviri")
n_lat, n_lon = grid.get_shape()

reader = open_store_reader("seviri", SEVIRI_STORE, ["data_daily"])
daily = reader.data_daily_arr

first_day = T1H.date_to_index(f"{YEAR}-01-01") // 24
last_day = T1H.date_to_index(f"{YEAR + 1}-01-01") // 24

n_pixels = daily.shape[0]
fvc_mean = np.full(n_pixels, np.nan, dtype=np.float32)
valid_days = np.zeros(n_pixels, dtype=np.int16)

for row_start in range(0, n_pixels, ROW_BLOCK):
    row_stop = min(row_start + ROW_BLOCK, n_pixels)
    block = daily[row_start:row_stop, first_day:last_day]["FVC"].squeeze(axis=(2, 3))

    is_valid = np.isfinite(block)
    day_count = is_valid.sum(axis=1)
    day_total = np.where(is_valid, block, 0.0).sum(axis=1)

    valid_days[row_start:row_stop] = day_count
    covered = day_count > 0
    fvc_mean[row_start:row_stop][covered] = day_total[covered] / day_count[covered]

fvc = fvc_mean.reshape(n_lat, n_lon)
days = valid_days.reshape(n_lat, n_lon)

# --- write ------------------------------------------------------------------

dataset = xr.Dataset(
    {
        "fvc": (("lat", "lon"), fvc),
        "valid_days": (("lat", "lon"), days),
    },
    coords={"lat": grid.y, "lon": grid.x},
)
dataset["fvc"].attrs = {
    "long_name": f"mean fractional vegetation cover, {YEAR}",
    "units": "1",
    "source": f"seviri full store {SEVIRI_STORE}, data_daily FVC",
    "note": "mean over days with a valid retrieval; NaN where no day was valid",
}
dataset["valid_days"].attrs = {
    # no "units" attribute: xarray decodes a units of "days" into timedelta64 on reload
    "long_name": f"number of days in {YEAR} contributing to the mean",
}
dataset.attrs = {"grid": grid.name, "resolution_deg": grid.resolution, "year": YEAR}
dataset.to_netcdf(OUT_PATH)

# --- visual check on the written file ---------------------------------------

field = xr.open_dataset(OUT_PATH)["fvc"]
fvc_flat = field.values.ravel()

cmap = plt.get_cmap("Greens").copy()
cmap.set_bad("0.9")

extent = (
    float(field.lon[0]) - grid.resolution / 2,
    float(field.lon[-1]) + grid.resolution / 2,
    float(field.lat[-1]) - grid.resolution / 2,
    float(field.lat[0]) + grid.resolution / 2,
)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

image = axes[0].imshow(field.values, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1)
axes[0].set_title(f"mean FVC {YEAR}, full seviri grid ({n_lat} x {n_lon})")
fig.colorbar(image, ax=axes[0], shrink=0.7, label="fractional vegetation cover")

axes[1].imshow(field.values, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1, alpha=0.35)
for region in EVAL_REGIONS:
    ids = np.sort(np.load(f"assets/spatial/indices_{region}.npz")["src_indices"].astype(np.int64))
    centers = grid.centers[ids]
    axes[1].scatter(centers[:, 0], centers[:, 1], c=fvc_flat[ids], cmap=cmap, vmin=0, vmax=1, s=0.5, marker="s")
axes[1].set_xlim(-25, 55)
axes[1].set_ylim(-38, 40)
axes[1].set_title("eval cells read via ravel()[src_indices], overlaid")

for ax in axes:
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

fig.tight_layout()
fig.savefig(PLOT_DIR / f"fvc_{YEAR}_check.png", dpi=95)

# --- distribution at the training sites -------------------------------------

anchor_ids = np.sort(np.load(INDICES_PATH)["src_indices"].astype(np.int64))
seviri_rows = map_to_store_rows(anchor_grid("mtg"), anchor_ids, store_grid("seviri"))

site_fvc = fvc_flat[seviri_rows]
valid = np.isfinite(site_fvc)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(site_fvc[valid], bins=50, color="#3b7a4b")
ax.set_xlabel(f"mean fractional vegetation cover, {YEAR}")
ax.set_ylabel("number of training sites")
ax.set_title(f"indices_mtg_new  (n={valid.sum()})")

fig.tight_layout()
fig.savefig(PLOT_DIR / f"train_sites_fvc_{YEAR}_hist.png", dpi=95)
