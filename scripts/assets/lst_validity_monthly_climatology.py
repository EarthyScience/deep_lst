"""Monthly LST observational validity (percent of valid hourly retrievals) on the full seviri grid.

Pools every hour of a calendar month across the years 2016-2024 and writes one NetCDF per
month with a single variable, lst_validity: the share of hourly slots with a clear-sky LST, in
percent. Hours without a source file count as invalid, so this is the availability the model
actually sees. Sibling of fvc_monthly_climatology.py.

LST is read from the hourly store seviri_product_hourly_lst, one inner chunk of the grid
(291 x 291 cells) and one calendar month at a time; the valid hours are counted per month and
summed over the years. Cells outside the land grid of the store have no value.

    python scripts/assets/lst_validity_monthly_climatology.py
    python scripts/assets/lst_validity_monthly_climatology.py --blocks 49 50 --out_dir /tmp/lst_smoke   # smoke
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.stores import T1H, get_data_reader
from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()
OUT_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/LSTvalidity_monthly_climatology")
LST_STORE = "seviri_product_hourly_lst"
FIRST_YEAR = 2016
LAST_YEAR = 2024


def calendar_months():
    """Every calendar month of the years as (month 1-12, first hour, hour after the last)."""
    starts = pd.date_range(f"{FIRST_YEAR}-01-01", f"{LAST_YEAR + 1}-01-01", freq="MS")
    pieces = []
    for start, stop in zip(starts[:-1], starts[1:]):
        pieces.append((start.month, T1H.date_to_index(start), T1H.date_to_index(stop)))
    return pieces


def block_rows(grid, lat_slice, lon_slice):
    """Grid rows of the cells of a rectangle, in the order of the flattened rectangle."""
    lat_index, lon_index = np.meshgrid(
        np.arange(lat_slice.start, lat_slice.stop), np.arange(lon_slice.start, lon_slice.stop), indexing="ij"
    )
    return grid.lat_lon_index_to_rows(lat_index.ravel(), lon_index.ravel())


def valid_hours_of_block(reader, lat_slice, lon_slice, pieces):
    """Number of hours with a valid LST per calendar month and cell of a rectangle. (12, cells)"""
    channel = reader.variables.index("LST")
    n_cells = (lat_slice.stop - lat_slice.start) * (lon_slice.stop - lon_slice.start)
    valid_hours = np.zeros((12, n_cells), dtype=np.int64)
    for month, first_hour, last_hour in pieces:
        box = reader.get_data_box(lat_slice, lon_slice, first_hour, last_hour)  # (hours, lat, lon, variables)
        is_valid = np.isfinite(box[..., channel])
        valid_hours[month - 1] += is_valid.sum(axis=0).ravel()
    return valid_hours


def write_month(month, validity, grid, out_dir):
    n_lat, n_lon = grid.shape
    dataset = xr.Dataset(
        {"lst_validity": (("lat", "lon"), validity.reshape(n_lat, n_lon))},
        coords={"lat": grid.lat, "lon": grid.lon},
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
        "resolution_deg": grid.lon_step,
        "source": f"store {LST_STORE}, variable LST",
    }
    out_path = out_dir / f"lst_validity_climatology_{month:02d}.nc"
    dataset.to_netcdf(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reader = get_data_reader(LST_STORE, PATHS.product_store_path(LST_STORE))
    grid = reader.spatial_grid
    rectangles = reader.spatial_chunks()
    pieces = calendar_months()
    total_hours = np.zeros(12, dtype=np.int64)
    for month, first_hour, last_hour in pieces:
        total_hours[month - 1] += last_hour - first_hour

    n_pixels = len(grid)
    first_block, last_block = (0, len(rectangles)) if args.blocks is None else args.blocks
    validity = np.full((12, n_pixels), np.nan, dtype=np.float32)
    print(
        f"blocks {first_block}-{last_block} of {len(rectangles)} x {len(pieces)} months ({FIRST_YEAR}-{LAST_YEAR})",
        flush=True,
    )

    for number in range(first_block, last_block):
        lat_slice, lon_slice = rectangles[number]
        valid_hours = valid_hours_of_block(reader, lat_slice, lon_slice, pieces)
        rows = block_rows(grid, lat_slice, lon_slice)
        validity[:, rows] = 100.0 * valid_hours / total_hours[:, None]
        observed = int((valid_hours.sum(axis=0) > 0).sum())
        print(f"  block {number}: {observed} pixels with observations", flush=True)

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
