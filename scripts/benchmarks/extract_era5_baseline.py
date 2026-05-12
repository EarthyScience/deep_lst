"""Extract ERA5 SKT baseline for an evaluation region.

Reads SKT from the preloaded ERA5 sampled eval zarr store
(era5_sampled_eval_{region}.zarr) and saves a NetCDF with dimensions
(valid_time, lat, lon) for dense eval regions or (valid_time, pixel) for
sparse training-site regions where the bounding-box grid would be too large.

The store already contains only the ERA5 pixels that cover the eval region,
so no spatial mapping is needed beyond remapping pixels to a lat/lon grid.
The full time axis of the store (2016–2024) is available; by default the
output covers the full test year.

The result is experiment-independent and shared across all model runs.
Re-run with --overwrite to refresh.

For training-site eval regions (e.g. train_KG_L_fullyear) the dedicated
eval zarr store does not exist — use the training-region store instead:

    python scripts/benchmarks/extract_era5_baseline.py \\
        --eval train_KG_L_fullyear \\
        --store_spatial KG_L \\
        --index_name KG_L

Usage
-----
python scripts/benchmarks/extract_era5_baseline.py --eval dahra
python scripts/benchmarks/extract_era5_baseline.py --eval dahra --test_year 2024

sbatch scripts/benchmarks/extract_era5_baseline.sh --eval dahra --test_year 2024
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

# Mirrors inference.py: pixel fill fraction below this → sparse (pixel) output.
_SPARSE_FILL_THRESHOLD = 0.01


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
    p.add_argument(
        "--store_spatial",
        default=None,
        help=(
            "Override the zarr store spatial key. Default: eval_{eval}. Use e.g. KG_L for training-site eval regions."
        ),
    )
    p.add_argument(
        "--index_name",
        default=None,
        help=(
            "Override the ERA5 spatial index name. "
            "Default: eval_{eval} → assets/era5/spatial/indices_eval_{eval}.npz. "
            "Use e.g. KG_L to point to indices_KG_L.npz."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline file")
    return p.parse_args()


def extract_baseline(
    eval_name: str,
    test_year: int,
    store_spatial: str | None = None,
    index_name: str | None = None,
) -> xr.Dataset:
    spatial_key = store_spatial if store_spatial is not None else f"eval_{eval_name}"
    store_path = PATHS.sampled_store_path("era5", spatial_key)
    if not store_path.exists():
        raise FileNotFoundError(
            f"ERA5 sampled store not found: {store_path}\n"
            "For training-site eval regions pass --store_spatial <key>, e.g. KG_L."
        )

    # --- Spatial mapping: SEVIRI pixels → ERA5 pixels ---
    # The output grid must match SEVIRI (same as the forecast), not ERA5 resolution.
    if index_name is not None:
        era5_idx_path = PATHS.spatial_indices_path("era5", index_name)
    else:
        era5_idx_path = PATHS.eval_indices_path("era5", eval_name)
    if not era5_idx_path.exists():
        raise FileNotFoundError(
            f"ERA5 spatial index not found: {era5_idx_path}\n"
            "For training-site eval regions pass --index_name <key>, e.g. KG_L."
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

    # --- Remap to SEVIRI spatial layout ---
    unique_seviri = np.unique(seviri_src_indices)
    lats, lons, local_rows, local_cols, grid_shape = build_local_grid(unique_seviri)

    # pixel_positions[i] = rank of seviri_src_indices[i] in unique_seviri
    _, pixel_positions = np.unique(seviri_src_indices, return_inverse=True)

    # skt_pixels[i, :] = SKT for unique_seviri[i] — shape (n_unique, n_test)
    skt_pixels = skt_seviri[pixel_positions, :]

    fill = len(unique_seviri) / (grid_shape[0] * grid_shape[1])
    sparse = fill < _SPARSE_FILL_THRESHOLD
    print(f"Spatial: {len(unique_seviri)} pixels, fill={fill:.4f} → {'sparse' if sparse else 'dense'} output")

    attrs = {
        "eval_region": eval_name,
        "test_year": str(test_year),
        "description": "ERA5 skin temperature baseline at SEVIRI eval pixel locations.",
        "units": "K",
    }

    if sparse:
        pixel_lats = lats[local_rows]
        pixel_lons = lons[local_cols]
        ds = xr.Dataset(
            {"skt": (["valid_time", "pixel"], skt_pixels.T)},
            coords={
                "valid_time": test_ts.values,
                "lat": ("pixel", pixel_lats),
                "lon": ("pixel", pixel_lons),
            },
            attrs=attrs,
        )
    else:
        skt_grid = np.full((n_test, *grid_shape), np.nan, dtype=np.float32)
        skt_grid[:, local_rows, local_cols] = skt_pixels.T
        ds = xr.Dataset(
            {"skt": (["valid_time", "lat", "lon"], skt_grid)},
            coords={
                "valid_time": test_ts.values,
                "lat": lats,
                "lon": lons,
            },
            attrs=attrs,
        )
    return ds


def main():
    args = parse_args()
    out_path = PATHS.baseline_path(args.eval)

    if out_path.exists() and not args.overwrite:
        print(f"Baseline already exists: {out_path}  (pass --overwrite to refresh)")
        return

    ds = extract_baseline(args.eval, args.test_year, args.store_spatial, args.index_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved ERA5 baseline to {out_path}")
    print(f"  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
    if "pixel" in ds.dims:
        print(f"  sparse: {ds.sizes['pixel']} pixels, {ds.sizes['valid_time']} timesteps")
    else:
        print(f"  dense: {ds.sizes['lat']}×{ds.sizes['lon']}, {ds.sizes['valid_time']} timesteps")


if __name__ == "__main__":
    main()
