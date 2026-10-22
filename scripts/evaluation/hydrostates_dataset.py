"""Long-format dataset of the hydrostates ladder for the mixed model (lme4), one CSV per eval.

One row per site x run x solar window x forecast day, joining the per-site verification
scores (scores/pixel_daily.nc, sample counts from verif/by_pixel.nc) with the site
attributes of the selection npz. Columns:

    site            MTG global cell id (src_indices); lon, lat of the cell centre
    in_training     site is also an mtg_new training site (20 of the 400 holdout sites)
    kg_class, lc_class, fvc_annual_mean, fvc_class, lst_validity   site attributes
    condition, seed, run                        data config, seed, "<condition>_s<seed>"
    window, day                                 day | night (| all), forecast day
    n_valid                                     valid (forecast, obs) samples pooled into the row
    rmse_model, bias_model, rmse_clim, rmse_persistence, rmse_era5land
    skill_clim                                  1 - rmse_model / rmse_clim
    log_rmse_model, log_rmse_clim               the response and its site offset

    python scripts/evaluation/hydrostates_dataset.py --eval_name eval_mtg
    python scripts/evaluation/hydrostates_dataset.py --eval_name eval_mtg --state_windows

The second form takes the 20 greenness-at-init windows (hydrostates_runs.GREEN_WINDOWS) instead
of day/night and writes hydrostates_<eval>_state_windows.csv next to the first file.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from eodata.selection.stratification import kg_major_from_classes  # noqa: E402
from hydrostates_runs import CONDITIONS, GREEN_WINDOWS, OUT_DIR, SEEDS, WINDOWS, run_name  # noqa: E402

from deep_lst.utils.io import ExperimentPaths  # noqa: E402
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid  # noqa: E402

PATHS = ExperimentPaths()
BASELINES = {"clim_smooth": "clim", "persistence": "persistence", "era5land": "era5land"}
# Global Aridity Index v3.1 (annual P/PET, 1970-2000) regridded to the seviri equiangular grid by
# deep_lst/preprocessing/aridity_index.py; a site climate, not a state.
ARIDITY_PATH = Path(
    "/Net/Groups/BGI/tscratch/weathergenerator/aux/AI_ET0__annual_v3_1/africa_ai_v31_yr_equiangular_005.nc"
)


def site_aridity(site):
    """Aridity index at each site's seviri pixel (NaN outside the regridded field)."""
    rows = map_to_store_rows(anchor_grid("mtg"), site, store_grid("seviri"))
    field = xr.open_dataset(ARIDITY_PATH)["aridity_index"].values.ravel()
    return field[rows]


def site_table(eval_name, training_selection="mtg_new"):
    """Per-site attributes in forecast pixel order (sorted src_indices); in_training is
    membership in training_selection."""
    npz = np.load(PATHS.spatial_indices_path(PATHS.baseline_spatial(eval_name)))
    order = np.argsort(npz["src_indices"], kind="stable")
    site = npz["src_indices"][order].astype(np.int64)
    centers = np.asarray(anchor_grid("mtg").centers)[site]
    training = np.load(PATHS.spatial_indices_path(training_selection))["src_indices"].astype(np.int64)
    kg_class = npz["kg_classes"][order]
    return pd.DataFrame(
        {
            "site": site,
            "lon": centers[:, 0],
            "lat": centers[:, 1],
            "in_training": np.isin(site, training),
            "kg_class": kg_class,
            "kg_major": kg_major_from_classes(kg_class.astype(np.int8)).astype(np.int16),  # 1-5 = A-E
            "lc_class": npz["modis_lc_classes"][order],
            "ecoregion": npz["ecoregion_classes"][order],
            "biome": npz["biome_classes"][order],
            "aridity": site_aridity(site),
            "elevation": npz["dem_values"][order],  # m, lisflood DEM sampled at the selection
            "fvc_annual_mean": npz["fvc_annual_mean"][order],
            "fvc_class": npz["fvc_classes"][order],
            "lst_validity": npz["lst_validity_annual_mean"][order],
        }
    )


def run_rows(eval_name, condition, seed, windows, days, sites):
    """Rows of one run, or None when its verification is missing."""
    results_dir = PATHS.results_dir(run_name(condition, seed), "version_0", eval_name)
    scores_path = results_dir / "scores" / "pixel_daily.nc"
    if not scores_path.exists():
        print(f"missing: {scores_path}")
        return None
    pixel_daily = xr.open_dataset(scores_path, decode_timedelta=False)
    # The verification carried the npz classes along as pixel coords: they must agree with the
    # site table row by row, otherwise the pixel order of this run differs from the npz order.
    if not np.array_equal(pixel_daily["fvc_class"].values, sites["fvc_class"].values):
        raise ValueError(f"pixel order of {scores_path} does not match the selection npz")
    counts = xr.open_dataset(results_dir / "verif" / "by_pixel.nc", decode_timedelta=False)["n"]
    day_of_lead = (counts["lead"] - 1) // 24 + 1

    frames = []
    for window in windows:
        for day in days:
            selected = pixel_daily.sel(window=window, day=day)
            rows = sites.copy()
            rows["condition"] = condition
            rows["seed"] = seed
            rows["run"] = f"{condition}_s{seed}"
            rows["window"] = window
            rows["day"] = day
            rows["n_valid"] = counts.sel(window=window).where(day_of_lead == day).sum("lead").values.astype(np.int64)
            rows["rmse_model"] = selected["rmse_model"].values
            rows["bias_model"] = selected["bias_model"].values
            for baseline, column in BASELINES.items():
                rows[f"rmse_{column}"] = selected["rmse_baseline"].sel(baseline=baseline).values
            rows["skill_clim"] = selected["rmse_ss"].sel(baseline="clim_smooth").values
            rows["log_rmse_model"] = np.log(rows["rmse_model"])
            rows["log_rmse_clim"] = np.log(rows["rmse_clim"])
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS))
    parser.add_argument(
        "--state_windows",
        action="store_true",
        help="the 20 greenness-at-init windows instead of --windows; output hydrostates_<eval>_state_windows.csv",
    )
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    windows = GREEN_WINDOWS if args.state_windows else args.windows

    sites = site_table(args.eval_name)
    frames = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows = run_rows(args.eval_name, condition, seed, windows, args.days, sites)
            if rows is not None:
                frames.append(rows)
    dataset = pd.concat(frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_state_windows" if args.state_windows else ""
    out_path = args.out_dir / f"hydrostates_{args.eval_name}{suffix}.csv"
    dataset.to_csv(out_path, index=False, float_format="%.6g")
    runs = dataset["run"].nunique()
    print(
        f"{out_path}: {len(dataset)} rows = {sites.shape[0]} sites x {runs} runs x {len(windows)} windows x {len(args.days)} days"
    )
    print(dataset.groupby(["condition", "seed"]).size().rename("rows").to_string())


if __name__ == "__main__":
    main()
