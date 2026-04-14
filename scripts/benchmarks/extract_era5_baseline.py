"""Extract ERA5 SKT baseline for an evaluation region.

Reads SKT from the preloaded ERA5 sampled eval zarr store
(era5_sampled_eval_{region}.zarr) and saves a gridded NetCDF with
dimensions (valid_time, lat, lon).

The store already contains only the ERA5 pixels that cover the eval region,
so no spatial mapping is needed beyond remapping pixels to a lat/lon grid.
The full time axis of the store (2016–2024) is available; by default the
output covers the full test year.

The result is experiment-independent and shared across all model runs.
Re-run with --overwrite to refresh.

Usage
-----
python scripts/benchmarks/extract_era5_baseline.py --eval dahra
python scripts/benchmarks/extract_era5_baseline.py --eval dahra --test_year 2024

sbatch scripts/benchmarks/extract_era5_baseline.sh
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval region name, e.g. dahra")
    p.add_argument(
        "--test_year",
        type=int,
        default=2024,
        help="Year to extract the baseline for (default: 2024).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline file")
    return p.parse_args()


def extract_baseline(eval_name: str, test_year: int) -> xr.Dataset:
    store_path = PATHS.sampled_store_path("era5", f"eval_{eval_name}")
    if not store_path.exists():
        raise FileNotFoundError(
            f"ERA5 sampled eval store not found: {store_path}\nRun the eval-store preloading step first."
        )

    # --- Spatial mapping: SEVIRI pixels → ERA5 pixels ---
    # The output grid must match SEVIRI (same as the forecast), not ERA5 resolution.
    era5_idx_path = PATHS.eval_indices_path("era5", eval_name)
    if not era5_idx_path.exists():
        raise FileNotFoundError(
            f"ERA5 eval index not found: {era5_idx_path}\nRun the spatial eval filtering script first."
        )
    idx_data = np.load(era5_idx_path)
    seviri_src_indices = idx_data["src_indices"]  # (n,) SEVIRI flat pixel indices
    era5_mapping = idx_data["mapping"]  # (n,) ERA5 flat pixel indices, aligned 1-to-1

    store = zarr.open(str(store_path), mode="r")

    era5_spatial_idx = store["spatial_idx"][:]  # ERA5 flat indices in store order
    timestamps = pd.DatetimeIndex(store["time_idx"][:])
    skt = store["skt"][:]  # (n_era5_pixels, n_time) float32

    print(f"Store  : {len(era5_spatial_idx)} ERA5 pixels × {len(timestamps)} timesteps")
    print(f"Range  : {timestamps[0]} → {timestamps[-1]}")

    test_mask = timestamps.year == test_year
    n_test = int(test_mask.sum())
    if n_test == 0:
        raise ValueError(f"No data found for test year {test_year} in store.")
    print(f"Test   : {n_test} timesteps  (year {test_year})")

    test_ts = timestamps[test_mask]
    skt_test = skt[:, test_mask]  # (n_era5_pixels, n_test)

    # --- Map ERA5 store ranks for each SEVIRI pixel ---
    era5_flat_to_rank = {int(v): i for i, v in enumerate(era5_spatial_idx)}
    seviri_to_era5_rank = np.array([era5_flat_to_rank[int(e)] for e in era5_mapping])

    # skt_seviri[i, :] = SKT for SEVIRI pixel seviri_src_indices[i]
    skt_seviri = skt_test[seviri_to_era5_rank, :]  # (n_seviri, n_test)

    # --- Remap to SEVIRI (lat, lon) grid ---
    unique_seviri = np.unique(seviri_src_indices)
    lats, lons, local_rows, local_cols, grid_shape = build_local_grid(unique_seviri)

    # pixel_positions[i] = rank of seviri_src_indices[i] in unique_seviri
    _, pixel_positions = np.unique(seviri_src_indices, return_inverse=True)

    skt_grid = np.full((n_test, *grid_shape), np.nan, dtype=np.float32)
    skt_grid[:, local_rows, local_cols] = skt_seviri[pixel_positions, :].T

    ds = xr.Dataset(
        {"skt": (["valid_time", "lat", "lon"], skt_grid)},
        coords={
            "valid_time": test_ts.values,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "eval_region": eval_name,
            "test_year": str(test_year),
            "description": "ERA5 skin temperature baseline at SEVIRI eval pixel locations.",
            "units": "K",
        },
    )
    return ds


def main():
    args = parse_args()
    out_path = PATHS.baseline_path(args.eval)

    if out_path.exists() and not args.overwrite:
        print(f"Baseline already exists: {out_path}  (pass --overwrite to refresh)")
        return

    ds = extract_baseline(args.eval, args.test_year)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved ERA5 baseline to {out_path}")
    print(f"  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
    print(f"  grid: {ds.sizes['lat']}×{ds.sizes['lon']}, {ds.sizes['valid_time']} timesteps")


if __name__ == "__main__":
    main()
