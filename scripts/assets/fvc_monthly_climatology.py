"""Monthly FVC climatology (mean, std, valid-day count) on the full seviri grid.

Pools every day of a calendar month across the store years (2016-2024) and writes one
NetCDF per month. Same full-store access and output layout as the former yearly-mean
script (preprocessing/vegetation_cover.py): FVC already lives on the seviri grid in
`data_daily`, so this is a temporal reduction only, read in row blocks.

    python scripts/assets/fvc_monthly_climatology.py
    python scripts/assets/fvc_monthly_climatology.py --rows 4096000 4100000 --out_dir /tmp/fvc_smoke   # smoke on a land block
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H

from deep_lst.utils.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.utils.spatial import store_grid

OUT_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/FVC_monthly_climatology")
FIRST_YEAR = 2016
LAST_YEAR = 2024
ROW_BLOCK = 100_000  # rows per read: 100k rows x 3288 days x float32 = 1.3 GB


def month_of_store_day(first_day, last_day):
    """Calendar month (1-12) of every store day in [first_day, last_day)."""
    dates = pd.date_range(f"{FIRST_YEAR}-01-01", periods=last_day - first_day, freq="D")
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


def write_month(month, mean, std, count, grid, out_dir):
    n_lat, n_lon = grid.get_shape()
    dataset = xr.Dataset(
        {
            "fvc_mean": (("lat", "lon"), mean.reshape(n_lat, n_lon)),
            "fvc_std": (("lat", "lon"), std.reshape(n_lat, n_lon)),
            "valid_days": (("lat", "lon"), count.reshape(n_lat, n_lon)),
        },
        coords={"lat": grid.y, "lon": grid.x},
    )
    dataset["fvc_mean"].attrs = {
        "long_name": f"mean fractional vegetation cover, month {month:02d}, {FIRST_YEAR}-{LAST_YEAR}",
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
        "years": f"{FIRST_YEAR}-{LAST_YEAR}",
        "grid": grid.name,
        "resolution_deg": grid.resolution,
        "source": f"seviri full store {SEVIRI_STORE}, data_daily FVC",
    }
    out_path = out_dir / f"fvc_climatology_{month:02d}.nc"
    dataset.to_netcdf(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--rows",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help="store row range to process instead of the whole grid (smoke test)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = store_grid("seviri")
    reader = open_store_reader("seviri", SEVIRI_STORE, ["data_daily"])
    daily = reader.data_daily_arr

    first_day = T1H.date_to_index(f"{FIRST_YEAR}-01-01") // 24
    last_day = T1H.date_to_index(f"{LAST_YEAR + 1}-01-01") // 24
    months = month_of_store_day(first_day, last_day)

    n_pixels = daily.shape[0]
    first_row, last_row = (0, n_pixels) if args.rows is None else args.rows
    mean = np.full((12, n_pixels), np.nan, dtype=np.float32)
    std = np.full((12, n_pixels), np.nan, dtype=np.float32)
    count = np.zeros((12, n_pixels), dtype=np.int16)
    print(
        f"rows {first_row}-{last_row} x {last_day - first_day} days ({FIRST_YEAR}-{LAST_YEAR}), blocks of {ROW_BLOCK}",
        flush=True,
    )

    for row_start in range(first_row, last_row, ROW_BLOCK):
        row_stop = min(row_start + ROW_BLOCK, last_row)
        block = daily[row_start:row_stop, first_day:last_day]["FVC"].squeeze(axis=(2, 3))
        block_mean, block_std, block_count = accumulate_block(block, months)
        mean[:, row_start:row_stop] = block_mean
        std[:, row_start:row_stop] = block_std
        count[:, row_start:row_stop] = block_count
        print(f"  rows {row_start}-{row_stop}: {int((block_count.sum(axis=0) > 0).sum())} pixels with data", flush=True)

    for month in range(1, 13):
        out_path = write_month(month, mean[month - 1], std[month - 1], count[month - 1], grid, args.out_dir)
        covered = int((count[month - 1] > 0).sum())
        print(
            f"month {month:02d}: {covered} pixels, mean FVC {np.nanmean(mean[month - 1]):.3f} -> {out_path}", flush=True
        )


if __name__ == "__main__":
    main()
