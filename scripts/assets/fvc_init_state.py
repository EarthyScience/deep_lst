"""Vegetation state at forecast init: relative greenness per site and day of the test year, on
the init day and over the model's lookback.

For every site of a selection, the store's daily FVC over the test year (plus the lookback
days before it) is read, cloud gaps are forward-filled up to --max_gap_days (or not at all
with --no_fill), and the value is expressed relative to the site's own annual cycle from the
FVC monthly climatology of the --clim_years:

    relative_greenness = (FVC(day) - min_month) / (max_month - min_month)

so 0 is the site at its climatological minimum and 1 at its maximum, comparable across sites
with opposite seasons. Variables, dims (site, time) of the test year:

    relative_greenness             init day, clipped to [0, 1] (the state windows, masks.GREEN_BINS)
    relative_greenness_unclipped   init day, unclipped
    relative_greenness_lookback    mean of the unclipped state over the --lookback_days days
                                   ending at the init day (the model's context window)
    relative_greenness_tendency    unclipped init day minus --lookback_days days earlier
    real_days                      observed (not carried) FVC days in that window
    fvc, carried_days              the filled daily FVC and how many days a value was carried

With --no_fill nothing is carried: the init-day state exists on observed days only, the
lookback mean averages the observed days of the window (at least one), the tendency needs
observed values at both ends; written as fvc_init_state_<selection>_observed.nc, the
sensitivity check of the gap filling. The printout reports the coverage per Köppen major class.

Written as assets/static/fvc_init_state_<selection>.nc; the verification attaches every
variable as an (init, pixel) coordinate (cubes.attach_init_state via common.setup_verif).

    python scripts/assets/fvc_init_state.py --selection eval_mtg_hydrostates_val
    python scripts/assets/fvc_init_state.py --selection eval_mtg_hydrostates_val --no_fill
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H
from eodata.selection.stratification import kg_major_from_classes

from deep_lst.utils.full_store import SEVIRI_STORE, open_store_reader
from deep_lst.utils.init_state import lookback_count, lookback_mean, lookback_tendency
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

PATHS = ExperimentPaths()
CLIMATOLOGY_ROOT = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived")
OUT_DIR = PATHS.assets_root / "static"
KG_MAJOR_NAMES = {1: "A tropical", 2: "B arid", 3: "C temperate", 4: "D continental", 5: "E polar"}


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


def climatology_range(rows, clim_years):
    """Per site, min and max of the twelve monthly climatological FVC means."""
    directory = CLIMATOLOGY_ROOT / f"FVC_monthly_climatology_{clim_years[0]}_{clim_years[1]}"
    monthly = np.stack(
        [
            xr.open_dataset(directory / f"fvc_climatology_{month:02d}.nc")["fvc_mean"].values.ravel()[rows]
            for month in range(1, 13)
        ]
    )
    return np.nanmin(monthly, axis=0), np.nanmax(monthly, axis=0), directory


def site_kg_major(npz):
    """Köppen major class (1-5 = A-E) per site in forecast pixel order (sorted src_indices)."""
    order = np.argsort(npz["src_indices"], kind="stable")
    return kg_major_from_classes(npz["kg_classes"][order].astype(np.int8)).astype(int)


def coverage_report(valid, kg_major, label):
    """Share of site-days with a value and of sites with a value on at least half of their
    days, per Köppen major class."""
    print(f"  {label}:")
    for code, name in KG_MAJOR_NAMES.items():
        sites = kg_major == code
        if not sites.any():
            continue
        site_days = valid[sites].mean()
        kept_sites = (valid[sites].mean(axis=1) >= 0.5).mean()
        print(
            f"    {name:14s} {sites.sum():5d} sites: {site_days:6.1%} of site-days, {kept_sites:6.1%} of sites with >= half their days"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default="eval_mtg")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--clim_years", type=int, nargs=2, default=[2016, 2023])
    parser.add_argument("--max_gap_days", type=int, default=15)
    parser.add_argument("--lookback_days", type=int, default=7)
    parser.add_argument(
        "--no_fill", action="store_true", help="observed days only, no carry-forward (sensitivity check)"
    )
    args = parser.parse_args()

    npz = np.load(PATHS.spatial_indices_path(args.selection))
    sites = np.sort(npz["src_indices"].astype(np.int64))  # forecast pixel order
    rows = map_to_store_rows(anchor_grid("mtg"), sites, store_grid("seviri"))
    kg_major = site_kg_major(npz)

    first_day = T1H.date_to_index(f"{args.year}-01-01") // 24 - args.lookback_days  # the window of 1 January
    last_day = T1H.date_to_index(f"{args.year + 1}-01-01") // 24
    times = pd.date_range(f"{args.year}-01-01", periods=last_day - first_day, freq="D") - pd.Timedelta(
        days=args.lookback_days
    )
    in_year = (times.year == args.year).to_numpy()

    reader = open_store_reader("seviri", SEVIRI_STORE, ["data_daily"])
    order = np.argsort(rows)  # zarr wants sorted rows; undo afterwards
    block = reader.data_daily_arr.oindex[rows[order], first_day:last_day]["FVC"].squeeze(axis=(2, 3))
    fvc = np.empty_like(block)
    fvc[order] = block
    fvc = fvc.astype(np.float32)

    site_min, site_max, climatology_dir = climatology_range(rows, args.clim_years)
    annual_range = site_max - site_min
    has_cycle = annual_range > 0

    valid_raw = np.isfinite(fvc)
    if args.no_fill:
        filled, carried = fvc.copy(), np.zeros(fvc.shape, dtype=np.int16)
    else:
        filled, carried = forward_fill(fvc, args.max_gap_days)
    unclipped = np.full(fvc.shape, np.nan, dtype=np.float32)
    unclipped[has_cycle] = (filled[has_cycle] - site_min[has_cycle, None]) / annual_range[has_cycle, None]
    min_days = 1 if args.no_fill else args.lookback_days
    lookback = lookback_mean(unclipped, args.lookback_days, min_days)
    tendency = lookback_tendency(unclipped, args.lookback_days)
    real_days = lookback_count(valid_raw, args.lookback_days)

    states = {
        "relative_greenness": np.clip(unclipped, 0.0, 1.0),
        "relative_greenness_unclipped": unclipped,
        "relative_greenness_lookback": lookback,
        "relative_greenness_tendency": tendency,
        "real_days": real_days,
        "fvc": filled,
        "carried_days": carried,
    }
    states = {name: values[:, in_year] for name, values in states.items()}
    valid_raw_year = valid_raw[:, in_year]

    variant = "observed days only" if args.no_fill else f"forward-filled up to {args.max_gap_days} days"
    print(
        f"{args.selection}: {len(sites)} sites x {int(in_year.sum())} days of {args.year}, {variant}, climatology {climatology_dir.name}"
    )
    longest_gap = np.array([longest_run(~site_valid) for site_valid in valid_raw_year])
    print(
        f"  raw FVC valid on {valid_raw_year.mean():.1%} of site-days; state valid on "
        f"{np.isfinite(states['relative_greenness']).mean():.1%}; longest gap per site: median {np.median(longest_gap):.0f} d, "
        f"90th percentile {np.percentile(longest_gap, 90):.0f} d, max {longest_gap.max()} d; "
        f"sites without an annual cycle: {(~has_cycle).sum()}"
    )
    for name in ("relative_greenness", "relative_greenness_lookback", "relative_greenness_tendency"):
        values = states[name]
        print(f"  {name}: deciles over valid site-days {np.round(np.nanquantile(values, np.linspace(0.1, 0.9, 9)), 2)}")
    coverage_report(np.isfinite(states["relative_greenness"]), kg_major, "init-day state")
    coverage_report(np.isfinite(states["relative_greenness_lookback"]), kg_major, "lookback mean")
    coverage_report(np.isfinite(states["relative_greenness_tendency"]), kg_major, "tendency")

    dataset = xr.Dataset(
        {name: (("site", "time"), values) for name, values in states.items()},
        coords={"site": sites, "time": times[in_year]},
    )
    notes = {
        "relative_greenness": "(fvc - min of the 12 monthly climatology means) / (max - min), clipped to [0, 1]",
        "relative_greenness_unclipped": "the same, unclipped",
        "relative_greenness_lookback": f"mean of the unclipped state over the {args.lookback_days} days ending at the day",
        "relative_greenness_tendency": f"unclipped state of the day minus {args.lookback_days} days earlier",
        "real_days": f"observed (not carried) FVC days in the {args.lookback_days}-day window",
        "fvc": "daily FVC, " + variant,
        "carried_days": "days the FVC value was carried forward from the last observation",
    }
    for name, note in notes.items():
        dataset[name].attrs = {"long_name": note, "missing": "NaN where FVC is missing or the site has no annual cycle"}
    dataset.attrs = {
        "selection": args.selection,
        "year": args.year,
        "climatology_years": f"{args.clim_years[0]}-{args.clim_years[1]}",
        "gap_fill": variant,
        "lookback_days": args.lookback_days,
        "store": SEVIRI_STORE,
        "climatology": str(climatology_dir),
        "generator": "scripts/assets/fvc_init_state.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_observed" if args.no_fill else ""
    out_path = OUT_DIR / f"fvc_init_state_{args.selection}{suffix}.nc"
    dataset.to_netcdf(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
