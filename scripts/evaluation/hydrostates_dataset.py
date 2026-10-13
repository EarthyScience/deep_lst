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
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.spatial import anchor_grid

PATHS = ExperimentPaths()
OUT_DIR = Path("results/independent/hydrostates")
CONDITIONS = ("full", "full_rzsm", "full_rzsm_dis", "full_glofas")
SEEDS = (42, 7, 23)
BASELINES = {"clim_smooth": "clim", "persistence": "persistence", "era5land": "era5land"}


def run_name(condition, seed):
    tag = "hydrostates" if seed == 42 else f"hydrostates_s{seed}"
    return f"mlp_smoothl1_h24_{condition}_v16_{tag}"


def site_table(eval_name):
    """Per-site attributes in forecast pixel order (sorted src_indices)."""
    npz = np.load(PATHS.spatial_indices_path(PATHS.baseline_spatial(eval_name)))
    order = np.argsort(npz["src_indices"], kind="stable")
    site = npz["src_indices"][order].astype(np.int64)
    centers = np.asarray(anchor_grid("mtg").centers)[site]
    training = np.load(PATHS.spatial_indices_path("mtg_new"))["src_indices"].astype(np.int64)
    return pd.DataFrame(
        {
            "site": site,
            "lon": centers[:, 0],
            "lat": centers[:, 1],
            "in_training": np.isin(site, training),
            "kg_class": npz["kg_classes"][order],
            "lc_class": npz["modis_lc_classes"][order],
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
    parser.add_argument("--windows", nargs="+", default=["day", "night"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sites = site_table(args.eval_name)
    frames = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows = run_rows(args.eval_name, condition, seed, args.windows, args.days, sites)
            if rows is not None:
                frames.append(rows)
    dataset = pd.concat(frames, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"hydrostates_{args.eval_name}.csv"
    dataset.to_csv(out_path, index=False, float_format="%.6g")
    runs = dataset["run"].nunique()
    print(
        f"{out_path}: {len(dataset)} rows = {sites.shape[0]} sites x {runs} runs x {len(args.windows)} windows x {len(args.days)} days"
    )
    print(dataset.groupby(["condition", "seed"]).size().rename("rows").to_string())


if __name__ == "__main__":
    main()
