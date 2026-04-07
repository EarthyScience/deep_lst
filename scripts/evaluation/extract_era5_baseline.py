"""Extract ERA5 SKT baseline for an evaluation region.

Reads ERA5 skin temperature at the SEVIRI eval pixel locations (via the
precomputed nearest-neighbour ERA5→SEVIRI mapping) and saves a gridded
NetCDF with dimensions (valid_time, lat, lon).

The result is experiment-independent and shared across all model runs.
Re-run with --overwrite to refresh (e.g. after adding more test periods).

Usage
-----
python scripts/evaluation/extract_era5_baseline.py \\
    --eval antananarivo \\
    --temporal test_period_00 test_period_03

sbatch scripts/evaluation/extract_era5_baseline.sh antananarivo test_period_00 test_period_03
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from eodata.data_reader.era5 import Era5Reader
from eodata.eo_zarr import EOZarrStore
from tqdm import tqdm

from deep_lst.evaluation.inference import TIME_INDEX_ORIGIN
from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval region name, e.g. antananarivo")
    p.add_argument(
        "--temporal",
        nargs="+",
        required=True,
        metavar="PERIOD",
        help="One or more temporal period names, e.g. test_period_00 test_period_03",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline file")
    p.add_argument(
        "--num_workers", type=int, default=0, help="Not used (sequential extraction); kept for interface consistency"
    )
    return p.parse_args()


def extract_baseline(eval_name: str, temporal_periods: list[str]) -> xr.Dataset:
    # --- Spatial indices ---
    # era5 eval file: src_indices = SEVIRI pixels, mapping = ERA5 pixels (1-to-1)
    era5_idx_path = PATHS.eval_indices_path("era5", eval_name)
    if not era5_idx_path.exists():
        raise FileNotFoundError(
            f"ERA5 eval index not found: {era5_idx_path}\nRun the spatial eval filtering script first."
        )
    idx_data = np.load(era5_idx_path)
    seviri_src_indices = idx_data["src_indices"]  # SEVIRI flat pixels — defines the output grid
    era5_mapping = idx_data["mapping"]  # ERA5 flat pixels — used for reading

    # --- Temporal indices ---
    time_idx = np.concatenate(
        [np.load(PATHS.temporal_indices_path(p))["time_indices"].astype(np.int64) for p in temporal_periods]
    )
    time_idx = np.sort(np.unique(time_idx))
    valid_times = TIME_INDEX_ORIGIN + pd.to_timedelta(time_idx, unit="h")

    # --- ERA5 reader (no normalisation, point-wise, skt only) ---
    store = EOZarrStore(PATHS.store_path("era5"), mode="r")
    reader = Era5Reader(
        eozarr_store=store,
        inputs=[],
        diagnostics=["skt"],
        prognostics=[],
        patch_size=1,
        lookback=0,
        horizon=0,
        row_extent=721,  # ERA5 0.25° global grid
        col_extent=1440,
    )
    # No normalisation — we want physical units (K)

    # --- Build output SEVIRI grid ---
    unique_seviri = np.unique(seviri_src_indices)
    lats, lons, local_rows, local_cols, grid_shape = build_local_grid(unique_seviri)

    n_time = len(time_idx)
    n_pixels = len(era5_mapping)

    # skt_flat: (n_pixels, n_time) — one ERA5 value per SEVIRI pixel per timestep
    skt_flat = np.full((n_pixels, n_time), np.nan, dtype=np.float32)

    # Group by unique ERA5 pixel to avoid redundant reads
    unique_era5, era5_to_pixels = np.unique(era5_mapping, return_inverse=True)

    for t_pos, t_idx in enumerate(tqdm(time_idx, desc=f"Extracting ERA5 SKT ({eval_name})")):
        for u_pos, era5_pix in enumerate(unique_era5):
            sample = reader.get_data_single(spatial_idx=int(era5_pix), time_idx=int(t_idx))
            val = sample["skt"].squeeze().item()
            # Assign to all SEVIRI pixels that map to this ERA5 pixel
            pixel_mask = era5_to_pixels == u_pos
            skt_flat[pixel_mask, t_pos] = val

    # --- Remap flat (n_pixels, n_time) → gridded (n_time, lat, lon) ---
    # seviri_src_indices and era5_mapping are aligned 1-to-1, ordered same as unique_seviri
    # We need position of each seviri_src_indices entry in unique_seviri
    _, pixel_positions = np.unique(seviri_src_indices, return_inverse=True)

    skt_grid = np.full((n_time, *grid_shape), np.nan, dtype=np.float32)
    skt_grid[:, local_rows, local_cols] = skt_flat[pixel_positions, :].T

    ds = xr.Dataset(
        {"skt": (["valid_time", "lat", "lon"], skt_grid)},
        coords={
            "valid_time": valid_times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "eval_region": eval_name,
            "temporal_periods": " ".join(temporal_periods),
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

    ds = extract_baseline(args.eval, args.temporal)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved ERA5 baseline to {out_path}")
    print(f"  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
    print(f"  grid: {ds.sizes['lat']}×{ds.sizes['lon']}, {ds.sizes['valid_time']} timesteps")


if __name__ == "__main__":
    main()
