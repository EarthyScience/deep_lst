"""Monthly FVC climatology (mean, std, valid-day count) on the full seviri grid.

Pools every day of a calendar month across --years and writes one NetCDF per month into
derived/FVC_monthly_climatology_<first>_<last>. The default 2016-2023 leaves the test year
out; it scales the init states (fvc_init_state.py). The 2016-2024 version in
derived/FVC_monthly_climatology is the site-selection raster and stays as it is.

FVC is read from the daily store seviri_product_daily_mdfvc, one inner chunk of the grid
(291 x 291 cells) after the other, so every chunk of the store is decoded once. Cells outside
the land grid of the store have no value.

    python scripts/assets/fvc_monthly_climatology.py --years 2016 2023
    python scripts/assets/fvc_monthly_climatology.py --blocks 49 50 --out_dir /tmp/fvc_smoke   # smoke on a land block
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.stores import T1D, get_data_reader
from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()
CLIMATOLOGY_ROOT = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived")
FVC_STORE = "seviri_product_daily_mdfvc"
DAYS_PER_READ = 366  # one block x 366 days x 3 variables x float32 = 0.4 GB


def month_of_store_day(first_day, last_day, first_year):
    """Calendar month (1-12) of every store day in [first_day, last_day)."""
    dates = pd.date_range(f"{first_year}-01-01", periods=last_day - first_day, freq="D")
    return dates.month.to_numpy()


def accumulate_block(block, months):
    """Per-month mean, std (ddof=1) and valid-day count of a (rows, days) block."""
    n_rows = block.shape[0]
    mean = np.full((12, n_rows), np.nan, dtype=np.float32)
    std = np.full((12, n_rows), np.nan, dtype=np.float32)
    count = np.zeros((12, n_rows), dtype=np.int16)
    for month in range(1, 13):
        values = block[:, months == month].astype(np.float64)
        is_valid = np.isfinite(values)
        day_count = is_valid.sum(axis=1)
        filled = np.where(is_valid, values, 0.0)
        day_sum = filled.sum(axis=1)
        day_sumsq = (filled * filled).sum(axis=1)

        has_mean = day_count > 0
        has_std = day_count > 1
        block_mean = np.full(n_rows, np.nan)
        block_mean[has_mean] = day_sum[has_mean] / day_count[has_mean]
        variance = np.full(n_rows, np.nan)
        variance[has_std] = (day_sumsq[has_std] - day_count[has_std] * block_mean[has_std] ** 2) / (
            day_count[has_std] - 1
        )
        variance = np.clip(variance, 0.0, None)  # round-off can push tiny variances below zero

        mean[month - 1] = block_mean
        std[month - 1] = np.sqrt(variance)
        count[month - 1] = day_count
    return mean, std, count


def block_rows(grid, lat_slice, lon_slice):
    """Grid rows of the cells of a rectangle, in the order of the flattened rectangle."""
    lat_index, lon_index = np.meshgrid(
        np.arange(lat_slice.start, lat_slice.stop), np.arange(lon_slice.start, lon_slice.stop), indexing="ij"
    )
    return grid.lat_lon_index_to_rows(lat_index.ravel(), lon_index.ravel())


def read_block(reader, lat_slice, lon_slice, first_day, last_day):
    """Daily FVC of a rectangle as (cells, days), NaN where missing."""
    channel = reader.variables.index("FVC")
    pieces = []
    for piece_start in range(first_day, last_day, DAYS_PER_READ):
        piece_stop = min(piece_start + DAYS_PER_READ, last_day)
        box = reader.get_data_box(lat_slice, lon_slice, piece_start, piece_stop)  # (days, lat, lon, variables)
        pieces.append(box[..., channel])
    fvc = np.concatenate(pieces)  # (days, lat, lon)
    return fvc.reshape(fvc.shape[0], -1).T


def write_month(month, mean, std, count, grid, out_dir, years):
    n_lat, n_lon = grid.shape
    dataset = xr.Dataset(
        {
            "fvc_mean": (("lat", "lon"), mean.reshape(n_lat, n_lon)),
            "fvc_std": (("lat", "lon"), std.reshape(n_lat, n_lon)),
            "valid_days": (("lat", "lon"), count.reshape(n_lat, n_lon)),
        },
        coords={"lat": grid.lat, "lon": grid.lon},
    )
    dataset["fvc_mean"].attrs = {
        "long_name": f"mean fractional vegetation cover, month {month:02d}, {years[0]}-{years[1]}",
        "units": "1",
        "note": "mean over all days of the month with a valid retrieval; NaN where none",
    }
    dataset["fvc_std"].attrs = {
        "long_name": f"standard deviation of daily fractional vegetation cover, month {month:02d}",
        "units": "1",
        "note": "sample std (ddof=1) over the same days; NaN where fewer than two",
    }
    dataset["valid_days"].attrs = {
        # no "units" attribute: xarray decodes a units of "days" into timedelta64 on reload
        "long_name": "number of days contributing to the statistics",
    }
    dataset.attrs = {
        "month": month,
        "years": f"{years[0]}-{years[1]}",
        "resolution_deg": grid.lon_step,
        "source": f"store {FVC_STORE}, variable FVC",
    }
    out_path = out_dir / f"fvc_climatology_{month:02d}.nc"
    dataset.to_netcdf(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs=2, default=[2016, 2023], metavar=("FIRST", "LAST"))
    parser.add_argument(
        "--out_dir", type=Path, default=None, help="default: derived/FVC_monthly_climatology_<first>_<last>"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="range of inner-chunk blocks to process instead of the whole grid (smoke test); "
        "blocks are numbered row by row, 11 per row",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = CLIMATOLOGY_ROOT / f"FVC_monthly_climatology_{args.years[0]}_{args.years[1]}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reader = get_data_reader(FVC_STORE, PATHS.product_store_path(FVC_STORE))
    grid = reader.spatial_grid
    rectangles = reader.spatial_chunks()

    first_day = T1D.date_to_index(f"{args.years[0]}-01-01")
    last_day = T1D.date_to_index(f"{args.years[1] + 1}-01-01")
    months = month_of_store_day(first_day, last_day, args.years[0])

    n_pixels = len(grid)
    first_block, last_block = (0, len(rectangles)) if args.blocks is None else args.blocks
    mean = np.full((12, n_pixels), np.nan, dtype=np.float32)
    std = np.full((12, n_pixels), np.nan, dtype=np.float32)
    count = np.zeros((12, n_pixels), dtype=np.int16)
    print(
        f"blocks {first_block}-{last_block} of {len(rectangles)} x {last_day - first_day} days ({args.years[0]}-{args.years[1]})",
        flush=True,
    )

    for number in range(first_block, last_block):
        lat_slice, lon_slice = rectangles[number]
        block = read_block(reader, lat_slice, lon_slice, first_day, last_day)
        block_mean, block_std, block_count = accumulate_block(block, months)
        rows = block_rows(grid, lat_slice, lon_slice)
        mean[:, rows] = block_mean
        std[:, rows] = block_std
        count[:, rows] = block_count
        print(f"  block {number}: {int((block_count.sum(axis=0) > 0).sum())} pixels with data", flush=True)

    for month in range(1, 13):
        out_path = write_month(month, mean[month - 1], std[month - 1], count[month - 1], grid, args.out_dir, args.years)
        covered = int((count[month - 1] > 0).sum())
        print(
            f"month {month:02d}: {covered} pixels, mean FVC {np.nanmean(mean[month - 1]):.3f} -> {out_path}", flush=True
        )


if __name__ == "__main__":
    main()
