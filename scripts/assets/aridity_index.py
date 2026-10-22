from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from eodata.util.spatial import get_admin_geometry
from matplotlib.colors import LogNorm
from rasterio.features import geometry_mask
from rasterio.windows import Window

from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

BASE = "/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/Global_Aridity_Index_and_PET_ET0_v3.1"
AI_PATH = f"{BASE}/Global-AI_ET0__annual_v3_1/ai_v31_yr.tif"
ET0_PATH = f"{BASE}/Global-AI_ET0__annual_v3_1/et0_v31_yr.tif"
OUT_DIR = Path("/Net/Groups/BGI/tscratch/weathergenerator/aux/AI_ET0__annual_v3_1")
BLOCK = 6
RESOLUTION = 0.05

INDICES_PATH = "deep_lst/assets/spatial/indices_sites_fvc.npz"


africa = get_admin_geometry(continent="Africa")
west, south, east, north = africa.geom.bounds

lon_centers = np.arange(np.floor(west / RESOLUTION), np.ceil(east / RESOLUTION) + 1) * RESOLUTION
lat_centers = np.arange(np.ceil(north / RESOLUTION), np.floor(south / RESOLUTION) - 1, -1) * RESOLUTION
n_cols, n_rows = len(lon_centers), len(lat_centers)

with rasterio.open(AI_PATH) as src:
    col_start = int(round((lon_centers[0] - RESOLUTION / 2 - src.transform.c) / src.transform.a))
    row_start = int(round((src.transform.f - (lat_centers[0] + RESOLUTION / 2)) / -src.transform.e))
    window = Window(col_start, row_start, n_cols * BLOCK, n_rows * BLOCK)
    ai_raw = src.read(1, window=window)

with rasterio.open(ET0_PATH) as src:
    et0_raw = src.read(1, window=window)

is_land = et0_raw != 65535
aridity_native = np.where(is_land, ai_raw * 1e-4, np.nan).astype(np.float32)

blocks = aridity_native.reshape(n_rows, BLOCK, n_cols, BLOCK)
with np.errstate(invalid="ignore"):
    aridity = np.nanmean(blocks, axis=(1, 3))

coarse_transform = rasterio.transform.from_origin(
    lon_centers[0] - RESOLUTION / 2, lat_centers[0] + RESOLUTION / 2, RESOLUTION, RESOLUTION
)
outside_africa = geometry_mask([africa.geom], out_shape=(n_rows, n_cols), transform=coarse_transform, invert=False)
aridity[outside_africa] = np.nan

extent = (
    lon_centers[0] - RESOLUTION / 2,
    lon_centers[-1] + RESOLUTION / 2,
    lat_centers[-1] - RESOLUTION / 2,
    lat_centers[0] + RESOLUTION / 2,
)

cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("0.9")
cmap.set_under("#8c6d3f")

fig, ax = plt.subplots(figsize=(8, 9))
image = ax.imshow(aridity, extent=extent, cmap=cmap, norm=LogNorm(vmin=0.003, vmax=3.0))
fig.colorbar(image, ax=ax, shrink=0.8, extend="both", label="aridity index (annual)")
fig.savefig("/Net/Groups/BGI/people/mwessel/deep_lst/results/aridity_index/africa_aridity.png", dpi=100)


# Resample and regrid AI index to match SEVIRI grid and save to file.
# save : dataset.to_netcdf(OUT_DIR / "africa_ai_v31_yr_equiangular_005.nc")

OCEAN_FLAG = 65535  # ET0 nodata; the AI file declares none of its own
AI_SCALE = 1e-4  # AI is stored as an integer with 4 decimal places
PIXELS_PER_CELL = 6  # one 0.05 deg cell = 6 x 6 native 30 arc-second pixels
RASTER_WEST = -180.0  # top-left corner of the native raster
RASTER_NORTH = 90.0
RASTER_PIXEL = 1 / 120  # 30 arc-seconds, in degrees

grid = store_grid("seviri")
n_lat, n_lon = grid.get_shape()
half_cell = grid.resolution / 2

first_col = round((grid.x[0] - half_cell - RASTER_WEST) / RASTER_PIXEL)
first_row = round((RASTER_NORTH - (grid.y[0] + half_cell)) / RASTER_PIXEL)

with rasterio.open(AI_PATH) as ai_src:
    covered_rows = (ai_src.height - first_row) // PIXELS_PER_CELL  # raster ends at 60 S
    window = Window(first_col, first_row, n_lon * PIXELS_PER_CELL, covered_rows * PIXELS_PER_CELL)
    ai_native = ai_src.read(1, window=window)

with rasterio.open(ET0_PATH) as et0_src:
    et0_native = et0_src.read(1, window=window)

is_land = et0_native != OCEAN_FLAG
ai_scaled = np.where(is_land, ai_native * AI_SCALE, np.nan).astype(np.float32)

blocks = ai_scaled.reshape(covered_rows, PIXELS_PER_CELL, n_lon, PIXELS_PER_CELL)
aridity = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
with np.errstate(invalid="ignore"):
    aridity[:covered_rows] = np.nanmean(blocks, axis=(1, 3))

dataset = xr.Dataset(
    {"aridity_index": (("lat", "lon"), aridity)},
    coords={"lat": grid.y, "lon": grid.x},
)
dataset.to_netcdf(OUT_DIR / "africa_ai_v31_yr_equiangular_005.nc")

field = xr.open_dataset(OUT_DIR / "africa_ai_v31_yr_equiangular_005.nc")["aridity_index"]
aridity_flat = field.values.ravel()
fig, ax = plt.subplots(figsize=(9, 9))
extent = (
    float(field.lon[0]) - 0.025,
    float(field.lon[-1]) + 0.025,
    float(field.lat[-1]) - 0.025,
    float(field.lat[0]) + 0.025,
)
ax.imshow(field.values, extent=extent, origin="upper", cmap=cmap, norm=LogNorm(vmin=0.003, vmax=3.0), alpha=0.35)

for region in ["eval_dahra", "eval_gobabeb", "eval_bouamir", "eval_yangambi", "eval_johannesburg"]:
    ids = np.sort(np.load(f"deep_lst/assets/spatial/indices_{region}.npz")["src_indices"].astype(np.int64))
    centers = grid.centers[ids]
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c=aridity_flat[ids],
        cmap=cmap,
        norm=LogNorm(vmin=0.003, vmax=3.0),
        s=0.5,
        marker="s",
    )

ax.set_xlim(-25, 55)
ax.set_ylim(-38, 40)
fig.savefig("/Net/Groups/BGI/people/mwessel/deep_lst/results/aridity_index/africa_aridity_valid.png", dpi=100)

# Plot distribution of aridity index values at training sites, with UNEP classification thresholds.

UNEP_EDGES = [0.03, 0.2, 0.5, 0.65]
UNEP_LABELS = ["hyper-arid", "arid", "semi-arid", "dry sub-humid", "humid"]

anchor_ids = np.sort(np.load(INDICES_PATH)["src_indices"].astype(np.int64))
seviri_rows = map_to_store_rows(anchor_grid("mtg"), anchor_ids, store_grid("seviri"))

site_aridity = aridity_flat[seviri_rows]
valid = np.isfinite(site_aridity)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(site_aridity[valid], bins=60, color="#3b6ea5")
axes[0].set_xlabel("aridity index")
axes[0].set_ylabel("number of training sites")
axes[0].set_title(f"indices_mtg_new  (n={valid.sum()})")

log_bins = np.logspace(np.log10(0.003), np.log10(4.0), 60)
axes[1].hist(np.clip(site_aridity[valid], 0.003, None), bins=log_bins, color="#3b6ea5")
axes[1].set_xscale("log")
axes[1].set_xlabel("aridity index (log scale)")
axes[1].set_title("log-spaced bins")

for ax in axes:
    for edge in UNEP_EDGES:
        ax.axvline(edge, color="0.4", lw=1, ls="--")

fig.tight_layout()
fig.savefig("/Net/Groups/BGI/people/mwessel/deep_lst/results/aridity_index/train_sites_aridity_hist.png", dpi=95)
