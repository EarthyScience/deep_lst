"""Hydrological state at forecast init: site-relative soil wetness per site and day of the test
year, on the init day and over the model's lookback.

Reads the preloaded input columns of an experiment's eval dataset, the same preload cache the
inference uses, so no store is touched. The cache holds normalised values, but a site-relative
range is invariant to the z-score (and monotone under the GloFAS signed-log1p transform, which
only compresses the wet end slightly). For every requested variable: daily mean of the 24
hourly values, the twelve monthly climatology means over the climatology years, and the daily
value relative to the site's climatological annual range, clipped to [0, 1]:

    wetness = (x(day) - min_month) / (max_month - min_month)

so 0 is the site at its climatological driest month and 1 at its wettest, comparable across
sites with opposite seasons, like the relative greenness of fvc_init_state.py. Three variables
per request, for the test year:

    <name>            the init day, clipped to [0, 1]
    <name>_unclipped  the init day, unclipped
    <name>_lookback   mean of the unclipped state over the --lookback_days days ending at the init
                      day: what the model saw in its context window (7 days = the 168 h lookback)
    <name>_tendency   unclipped init day minus --lookback_days days earlier: drying < 0 < wetting

Written as assets/static/hydro_init_state_<selection>.nc with dims (site, time); the
verification attaches every variable as an (init, pixel) coordinate (cubes.attach_init_state
via common.setup_verif), so they reach by_cell.nc and the init-level tables.

    python scripts/assets/hydro_init_state.py --selection eval_mtg \\
        --experiment mlp_smoothl1_h24_full_rzsm_v16_hydrostates \\
        --variables era5_land.swvl1:wetness glofas_v4.swir:rzsm_wetness

The experiment only selects the preload cache (its modalities and load lists); run it where
that cache exists, otherwise preload_shared starts building one.
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

from deep_lst.dataset.dataset import MultimodalDataset
from deep_lst.sensitivity.perturb import HOURS_PER_DAY, locate, preload_dates
from deep_lst.utils.init_state import lookback_mean, lookback_tendency
from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()
OUT_DIR = PATHS.assets_root / "static"


def daily_means(series):
    """(cells, hours) -> (cells, days) mean of the 24 hourly values, NaN where every hour is."""
    cells, hours = series.shape
    days = hours // HOURS_PER_DAY
    daily = series[:, : days * HOURS_PER_DAY].reshape(cells, days, HOURS_PER_DAY)
    with np.errstate(invalid="ignore"):
        return np.nanmean(daily, axis=2)


def monthly_range(daily, dates, years):
    """Per cell, min and max of the twelve monthly means over the climatology years."""
    in_years = (dates.year >= years[0]) & (dates.year <= years[1])
    monthly = np.stack([np.nanmean(daily[:, in_years & (dates.month == month)], axis=1) for month in range(1, 13)])
    return np.nanmin(monthly, axis=0), np.nanmax(monthly, axis=0)


def relative_state(daily, dates, years):
    """Daily values relative to the climatological annual range of monthly means, unclipped, on
    the whole day axis; NaN where the cell has no range."""
    site_min, site_max = monthly_range(daily, dates, years)
    annual_range = site_max - site_min
    has_range = np.isfinite(annual_range) & (annual_range > 0)
    relative = np.full(daily.shape, np.nan, dtype=np.float32)
    relative[has_range] = (daily[has_range] - site_min[has_range, None]) / annual_range[has_range, None]
    return relative, has_range


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default="eval_mtg")
    parser.add_argument("--experiment", default="mlp_smoothl1_h24_full_rzsm_v16_hydrostates")
    parser.add_argument("--version", default="version_0")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["era5_land.swvl1:wetness", "glofas_v4.swir:rzsm_wetness"],
        help="<modality>.<variable>:<state name>",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--clim_years", type=int, nargs=2, default=[2016, 2023])
    parser.add_argument("--lookback_days", type=int, default=7, help="window of <name>_lookback / <name>_tendency")
    args = parser.parse_args()

    config = OmegaConf.to_container(
        OmegaConf.load(PATHS.experiment_dir(args.experiment) / args.version / "config.yml"), resolve=True
    )
    dataset_config = dict(config["test_dataset"], spatial=args.selection)
    shared = MultimodalDataset.preload_shared(config["modality_ids"], **dataset_config)
    sites = np.sort(np.load(PATHS.spatial_indices_path(args.selection))["src_indices"].astype(np.int64))
    if not np.array_equal(shared["anchor_ids"], sites):
        raise ValueError("preload anchors are not the sorted src_indices of the selection")
    dates = preload_dates()
    times = pd.DatetimeIndex(dates[dates.year == args.year])

    in_year = dates.year == args.year
    states = {}
    for spec in args.variables:
        source, name = spec.split(":")
        modality, variable = source.split(".", 1)
        leaf, channel = locate(shared["var_names"], modality, variable)
        series = shared["columns"][modality][leaf][:, :, channel].numpy()
        relative, has_range = relative_state(daily_means(series), dates, args.clim_years)
        per_cell = {
            name: np.clip(relative, 0.0, 1.0),
            f"{name}_unclipped": relative,
            f"{name}_lookback": lookback_mean(relative, args.lookback_days),
            f"{name}_tendency": lookback_tendency(relative, args.lookback_days),
        }
        for state_name, values in per_cell.items():
            per_site = values[:, in_year][shared["inverse"][modality]]  # native cells -> sites
            states[state_name] = (("site", "time"), per_site)
            print(
                f"{state_name} ({modality}.{variable}): {len(sites)} sites on {int(has_range.sum())} of "
                f"{len(has_range)} native cells with a range; valid on {np.isfinite(per_site).mean():.1%} of "
                f"site-days; deciles over valid site-days "
                f"{np.round(np.nanquantile(per_site, np.linspace(0.1, 0.9, 9)), 2)}"
            )

    dataset = xr.Dataset(states, coords={"site": sites, "time": times})
    kinds = {
        "": "daily mean relative to the site's climatological annual range of monthly means, clipped to [0, 1]",
        "_unclipped": "the same, unclipped",
        "_lookback": f"mean of the unclipped relative state over the {args.lookback_days} days ending at the day",
        "_tendency": f"unclipped relative state minus its value {args.lookback_days} days earlier",
    }
    for state_name in states:
        suffix = next(kind for kind in ("_unclipped", "_lookback", "_tendency", "") if state_name.endswith(kind))
        dataset[state_name].attrs = {
            "long_name": f"{state_name}: {kinds[suffix]}",
            "note": "(x - min_month) / (max_month - min_month); NaN where the site has no range",
        }
    dataset.attrs = {
        "selection": args.selection,
        "year": args.year,
        "climatology_years": f"{args.clim_years[0]}-{args.clim_years[1]}",
        "lookback_days": args.lookback_days,
        "preload_of": f"{args.experiment}/{args.version}",
        "variables": " ".join(args.variables),
        "generator": "scripts/assets/hydro_init_state.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"hydro_init_state_{args.selection}.nc"
    dataset.to_netcdf(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
