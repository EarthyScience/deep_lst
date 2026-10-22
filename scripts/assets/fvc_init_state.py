"""Vegetation state at forecast init: relative greenness per site and day of the test year.

For every site of a selection, the store's daily FVC over the test year is read, cloud gaps
are forward-filled up to --max_gap_days, and the value is expressed relative to the site's
own annual cycle from the FVC monthly climatology:

    relative_greenness = (FVC(day) - min_month) / (max_month - min_month), clipped to [0, 1]

so 0 is the site at its climatological minimum and 1 at its maximum, comparable across
sites with opposite seasons. Written as assets/static/fvc_init_state_<selection>.nc with
dims (site, time); the verification attaches it as an (init, pixel) coordinate
(cubes.attach_init_state) and bins it into the state windows (masks.GREEN_BINS).

    python scripts/assets/fvc_init_state.py --selection eval_mtg
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H

from deep_lst.utils.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

PATHS = ExperimentPaths()
CLIMATOLOGY_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/FVC_monthly_climatology")
OUT_DIR = PATHS.assets_root / "static"


def forward_fill(values, max_gap):
    """Carry the last valid value forward along the time axis (axis 1) for at most max_gap
    days. Returns the filled values and, per cell, how many days the value was carried."""
    filled = values.copy()
    carried = np.zeros(values.shape, dtype=np.int16)
    for day in range(1, values.shape[1]):
        gap = ~np.isfinite(filled[:, day]) & np.isfinite(filled[:, day - 1]) & (carried[:, day - 1] < max_gap)
        filled[gap, day] = filled[gap, day - 1]
        carried[gap, day] = carried[gap, day - 1] + 1
    return filled, carried


def longest_run(flags):
    """Length of the longest run of True in a 1-D boolean array."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default="eval_mtg")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max_gap_days", type=int, default=15)
    args = parser.parse_args()

    npz = np.load(PATHS.spatial_indices_path(args.selection))
    sites = np.sort(npz["src_indices"].astype(np.int64))  # forecast pixel order
    rows = map_to_store_rows(anchor_grid("mtg"), sites, store_grid("seviri"))

    first_day = T1H.date_to_index(f"{args.year}-01-01") // 24
    last_day = T1H.date_to_index(f"{args.year + 1}-01-01") // 24
    times = pd.date_range(f"{args.year}-01-01", periods=last_day - first_day, freq="D")

    reader = open_store_reader("seviri", SEVIRI_STORE, ["data_daily"])
    order = np.argsort(rows)  # zarr wants sorted rows; undo afterwards
    block = reader.data_daily_arr.oindex[rows[order], first_day:last_day]["FVC"].squeeze(axis=(2, 3))
    fvc = np.empty_like(block)
    fvc[order] = block
    fvc = fvc.astype(np.float32)

    monthly = np.stack(
        [
            xr.open_dataset(CLIMATOLOGY_DIR / f"fvc_climatology_{month:02d}.nc")["fvc_mean"].values.ravel()[rows]
            for month in range(1, 13)
        ]
    )
    site_min, site_max = np.nanmin(monthly, axis=0), np.nanmax(monthly, axis=0)
    annual_range = site_max - site_min

    valid_raw = np.isfinite(fvc)
    filled, carried = forward_fill(fvc, args.max_gap_days)
    valid_filled = np.isfinite(filled)
    has_cycle = annual_range > 0
    greenness = np.full(fvc.shape, np.nan, dtype=np.float32)
    greenness[has_cycle] = np.clip(
        (filled[has_cycle] - site_min[has_cycle, None]) / annual_range[has_cycle, None], 0.0, 1.0
    )

    longest_gap = np.array([longest_run(~site_valid) for site_valid in valid_raw])
    print(f"{args.selection}: {len(sites)} sites x {len(times)} days of {args.year}")
    print(
        f"  raw FVC valid on {valid_raw.mean():.1%} of site-days; after forward-fill (max {args.max_gap_days} d): {valid_filled.mean():.1%}"
    )
    print(
        f"  longest gap per site: median {np.median(longest_gap):.0f} d, 90th percentile {np.percentile(longest_gap, 90):.0f} d, max {longest_gap.max()} d"
    )
    print(
        f"  sites without an annual cycle (range 0): {(~has_cycle).sum()} | relative greenness deciles over valid site-days: "
        f"{np.round(np.nanquantile(greenness, np.linspace(0.1, 0.9, 9)), 2)}"
    )

    dataset = xr.Dataset(
        {
            "relative_greenness": (("site", "time"), greenness),
            "fvc": (("site", "time"), filled),
            "carried_days": (("site", "time"), carried),
        },
        coords={"site": sites, "time": times},
    )
    dataset["relative_greenness"].attrs = {
        "long_name": "FVC relative to the site's climatological annual range",
        "note": "(fvc - min of the 12 monthly climatology means) / (max - min), clipped to [0, 1]; NaN where FVC is missing beyond max_gap_days or the site has no annual cycle",
    }
    dataset["fvc"].attrs = {"long_name": "daily FVC, cloud gaps forward-filled", "max_gap_days": args.max_gap_days}
    dataset["carried_days"].attrs = {"long_name": "days the FVC value was carried forward from the last observation"}
    dataset.attrs = {
        "selection": args.selection,
        "year": args.year,
        "store": SEVIRI_STORE,
        "climatology": str(CLIMATOLOGY_DIR),
        "generator": "scripts/assets/fvc_init_state.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"fvc_init_state_{args.selection}.nc"
    dataset.to_netcdf(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
