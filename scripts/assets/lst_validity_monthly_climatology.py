"""Monthly LST observational validity (percent of valid hourly retrievals) on the full seviri grid.

Pools every hour of a calendar month across the store years (2016-2024) and writes one
NetCDF per month with a single variable, lst_validity: the share of hourly slots with a
finite clear-sky LST, in percent. Hours without a source file count as invalid, so this is
the availability the model actually sees. Sibling of fvc_monthly_climatology.py; the hourly
leaf is 24x larger than the daily one, hence the smaller row block.

    python scripts/assets/lst_validity_monthly_climatology.py
    python scripts/assets/lst_validity_monthly_climatology.py --rows 4096000 4100000 --out_dir /tmp/lst_smoke   # smoke
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H

from deep_lst.exploration.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.utils.spatial import store_grid

OUT_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/LSTvalidity_monthly_climatology")
FIRST_YEAR = 2016
LAST_YEAR = 2024
ROW_BLOCK = 10_000  # rows per read: 10k rows x 3288 days x 24 h x float32 = 3.2 GB


def month_of_store_day(first_day, last_day):
    """Calendar month (1-12) of every store day in [first_day, last_day)."""
    dates = pd.date_range(f"{FIRST_YEAR}-01-01", periods=last_day - first_day, freq="D")
    return dates.month.to_numpy()


def validity_block(block, months):
    """Percent of valid hours per month of a (rows, days, 24) block."""
    is_valid = np.isfinite(block)
    validity = np.full((12, block.shape[0]), np.nan, dtype=np.float32)
    for month in range(1, 13):
        in_month = months == month
        valid_hours = is_valid[:, in_month, :].sum(axis=(1, 2))
        total_hours = in_month.sum() * 24
        validity[month - 1] = 100.0 * valid_hours / total_hours
    return validity


def write_month(month, validity, grid, out_dir):
    n_lat, n_lon = grid.get_shape()
    dataset = xr.Dataset(
        {"lst_validity": (("lat", "lon"), validity.reshape(n_lat, n_lon))},
        coords={"lat": grid.y, "lon": grid.x},
    )
    dataset["lst_validity"].attrs = {
        "long_name": f"share of hours with a valid clear-sky LST retrieval, month {month:02d}, {FIRST_YEAR}-{LAST_YEAR}",
        "units": "percent",
        "note": "valid hours / all hours of the month pooled over the years; hours without a source file "
        "count as invalid; NaN where the pixel never had a valid retrieval in any month",
    }
    dataset.attrs = {
        "month": month,
        "years": f"{FIRST_YEAR}-{LAST_YEAR}",
        "grid": grid.name,
        "resolution_deg": grid.resolution,
        "source": f"seviri full store {SEVIRI_STORE}, data LST",
    }
    out_path = out_dir / f"lst_validity_climatology_{month:02d}.nc"
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
    reader = open_store_reader("seviri", SEVIRI_STORE, ["data"])
    hourly = reader.data_arr

    first_day = T1H.date_to_index(f"{FIRST_YEAR}-01-01") // 24
    last_day = T1H.date_to_index(f"{LAST_YEAR + 1}-01-01") // 24
    months = month_of_store_day(first_day, last_day)

    n_pixels = hourly.shape[0]
    first_row, last_row = (0, n_pixels) if args.rows is None else args.rows
    validity = np.full((12, n_pixels), np.nan, dtype=np.float32)
    print(
        f"rows {first_row}-{last_row} x {last_day - first_day} days x 24 h ({FIRST_YEAR}-{LAST_YEAR}), blocks of {ROW_BLOCK}",
        flush=True,
    )

    for row_start in range(first_row, last_row, ROW_BLOCK):
        row_stop = min(row_start + ROW_BLOCK, last_row)
        block = hourly[row_start:row_stop, first_day:last_day]["LST"].squeeze(axis=(3, 4))  # (rows, days, 24)
        validity[:, row_start:row_stop] = validity_block(block, months)
        observed = int((validity[:, row_start:row_stop].max(axis=0) > 0).sum())
        print(f"  rows {row_start}-{row_stop}: {observed} pixels with observations", flush=True)

    never_observed = validity.max(axis=0) == 0  # water, off-disk, outside coverage: NaN rather than 0 %
    validity[:, never_observed] = np.nan

    for month in range(1, 13):
        out_path = write_month(month, validity[month - 1], grid, args.out_dir)
        observed = validity[month - 1] > 0
        print(
            f"month {month:02d}: {int(observed.sum())} pixels, mean validity {validity[month - 1][observed].mean():.1f} % -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
