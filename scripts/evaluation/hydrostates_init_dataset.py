"""Init-level dataset of the `full` (no GloFAS) runs on the holdout, for the within-site model
of skill against the vegetation state and the observational validity at init (lme4).

One row per site x init x solar window x forecast day, from verif/by_cell.nc (verif_cell.py):

    site, lon, lat, in_training, kg_class, lc_class, fvc_annual_mean, fvc_class, lst_validity
    seed, run, init, window, day
    greenness        FVC on the init day relative to the site's climatological annual range (0-1)
    coverage         share of valid LST hours in the lookback window before the init (0-1)
    n_valid          jointly valid (forecast, obs) hours pooled into the row (of 24)
    rmse_model, rmse_clim, skill_clim, log_rmse_model, log_rmse_clim

Inits default to 00 UTC only (greenness is a daily state; four inits a day would quadruple
the rows with the same state), windows day/night, days 1 and 10.

    python scripts/evaluation/hydrostates_init_dataset.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from hydrostates_dataset import run_name, site_table  # noqa: E402

from deep_lst.utils.io import ExperimentPaths  # noqa: E402

PATHS = ExperimentPaths()
OUT_DIR = Path("results/independent/hydrostates")
CONDITION = "full"
SEEDS = (42, 7, 23)


def seed_rows(eval_name, seed, windows, days, init_hours, sites):
    path = PATHS.results_dir(run_name(CONDITION, seed), "version_0", eval_name) / "verif" / "by_cell.nc"
    if not path.exists():
        print(f"missing: {path}")
        return None
    cell = xr.open_dataset(path, decode_timedelta=False)
    init_hour = pd.DatetimeIndex(cell["init"].values).hour
    cell = cell.isel(init=np.isin(init_hour, init_hours))
    n_sites, n_inits = cell.sizes["pixel"], cell.sizes["init"]
    frames = []
    for window in windows:
        for day in days:
            selected = cell.sel(window=window, day=day)
            rows = pd.DataFrame(np.repeat(sites.values, n_inits, axis=0), columns=sites.columns)
            rows["seed"] = seed
            rows["run"] = f"{CONDITION}_s{seed}"
            rows["init"] = np.tile(cell["init"].values, n_sites)
            rows["window"] = window
            rows["day"] = day
            rows["greenness"] = cell["relative_greenness"].values.T.ravel()  # (init, pixel) -> pixel-major
            rows["coverage"] = cell["lookback_coverage"].values.T.ravel()
            rows["n_valid"] = selected["n"].values.T.ravel().astype(np.int64)
            rows["rmse_model"] = np.sqrt(selected["mse_model"].values.T.ravel())
            rows["rmse_clim"] = np.sqrt(selected["mse_clim"].values.T.ravel())
            rows["skill_clim"] = 1.0 - rows["rmse_model"] / rows["rmse_clim"]
            rows["log_rmse_model"] = np.log(rows["rmse_model"])
            rows["log_rmse_clim"] = np.log(rows["rmse_clim"])
            frames.append(rows[rows["n_valid"] > 0])
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--windows", nargs="+", default=["day", "night"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--init_hours", type=int, nargs="+", default=[0])
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    sites = site_table(args.eval_name)
    frames = [seed_rows(args.eval_name, seed, args.windows, args.days, args.init_hours, sites) for seed in SEEDS]
    dataset = pd.concat([f for f in frames if f is not None], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"hydrostates_{args.eval_name}_init_level.csv"
    dataset.to_csv(out_path, index=False, float_format="%.6g", date_format="%Y-%m-%dT%H")
    print(
        f"{out_path}: {len(dataset)} rows | seeds {sorted(dataset.seed.unique())} | greenness valid {dataset.greenness.notna().mean():.1%} | coverage valid {dataset.coverage.notna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
