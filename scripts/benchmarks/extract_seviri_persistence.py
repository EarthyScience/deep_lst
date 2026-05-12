"""Extract SEVIRI LST persistence baseline for an eval region.

Usage
-----
uv run scripts/benchmarks/extract_seviri_persistence.py --eval train_MT --store MT
python scripts/benchmarks/extract_seviri_persistence.py --eval dahra \\
    --store KG_L

sbatch scripts/benchmarks/extract_seviri_persistence.sh dahra
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()

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
        help="Year to extract observations for (default: 2024).",
    )
    p.add_argument(
        "--store",
        default=None,
        help=(
            "Override the zarr store spatial key. Default: eval_{eval}. Use e.g. KG_L for training-site eval regions."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    return p.parse_args()


def extract_persistence(
    eval_name: str,
    test_year: int,
    store: str | None = None,
) -> xr.Dataset:
    spatial_key = store if store is not None else f"eval_{eval_name}"
    store_path = PATHS.sampled_store_path("seviri", spatial_key)

    store = zarr.open(str(store_path), mode="r")

    spatial_idx = store["spatial_idx"][:]
    timestamps = pd.DatetimeIndex(store["time_idx"][:])
    lst = store["LST"][:]  # (n_pixels, n_time) float32

    n_pixels = lst.shape[0]
    print(f"Store  : {n_pixels} pixels × {len(timestamps)} timesteps")
    print(f"Range  : {timestamps[0]} → {timestamps[-1]}")

    test_mask = timestamps.year == test_year
    n_test = int(test_mask.sum())

    test_ts = timestamps[test_mask]
    test_lst = lst[:, test_mask].T.astype(np.float32)  # (n_time, n_pixels)

    lats, lons, local_rows, local_cols, grid_shape = build_local_grid(spatial_idx)
    fill = n_pixels / (grid_shape[0] * grid_shape[1])
    sparse = fill < _SPARSE_FILL_THRESHOLD
    print(f"Spatial: {n_pixels} pixels, fill={fill:.4f} → {'sparse' if sparse else 'dense'} output")

    attrs = dict(
        eval_region=eval_name,
        test_year=str(test_year),
        units="K",
        description=(
            "SEVIRI LST observations for the test year. "
            "Persistence forecast at valid_time t uses obs(t - lead_time). "
            "run_evaluation.py applies the lead_time shift during alignment."
        ),
    )

    if sparse:
        if "lat" in store:
            pixel_lats = store["lat"][:]
            pixel_lons = store["lon"][:]
        else:
            pixel_lats = lats[local_rows]
            pixel_lons = lons[local_cols]
        return xr.Dataset(
            {"LST": (["valid_time", "pixel"], test_lst)},
            coords={
                "valid_time": test_ts.values,
                "lat": ("pixel", pixel_lats),
                "lon": ("pixel", pixel_lons),
            },
            attrs=attrs,
        )
    else:
        out = np.full((n_test, *grid_shape), np.nan, dtype=np.float32)
        out[:, local_rows, local_cols] = test_lst
        return xr.Dataset(
            {"LST": (["valid_time", "lat", "lon"], out)},
            coords={"valid_time": test_ts.values, "lat": lats, "lon": lons},
            attrs=attrs,
        )


def main():
    args = parse_args()
    out_path = PATHS.persistence_path(args.eval)

    if out_path.exists() and not args.overwrite:
        print(f"File already exists: {out_path}  (pass --overwrite to refresh)")
        return

    ds = extract_persistence(args.eval, args.test_year, args.store)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"Saved: {out_path}")
    print(f"  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
    if "pixel" in ds.dims:
        print(f"  sparse: {ds.sizes['pixel']} pixels, {ds.sizes['valid_time']} timesteps")
    else:
        print(f"  dense: {ds.sizes['lat']}×{ds.sizes['lon']}, {ds.sizes['valid_time']} timesteps")


if __name__ == "__main__":
    main()
