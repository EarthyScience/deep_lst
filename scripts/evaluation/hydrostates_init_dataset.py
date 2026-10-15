"""Init-level dataset of the hydrostates runs for the mixed models (lme4): one CSV per eval.

One row per site x init x solar window x forecast day x run, from verif/by_cell.nc
(verif_cell.py):

    site, lon, lat, in_training, kg_class, lc_class, ecoregion, biome, aridity, fvc_annual_mean,
    fvc_class, lst_validity                      site attributes (aridity = Global Aridity Index v3.1)
    site_set         holdout (eval_mtg) | training (train_mtg_new), for stacking the two files
    condition, seed, run, init, window, day      window = solar window (day/night) or solar-hour bin (h00_04 ...)
    greenness        FVC on the init day relative to the site's climatological annual range (0-1)
    wetness          ERA5-Land swvl1 on the init day relative to the site's climatological annual range (0-1)
    rzsm_wetness     the same for the GloFAS root-zone soil wetness index (where the asset has it)
    coverage         share of valid LST hours in the lookback window before the init (0-1)
    n_valid          jointly valid (forecast, obs) hours pooled into the row
    rmse_model, bias_model, rmse_clim, rmse_persistence, rmse_era5land   over the model's samples
    skill_clim, log_rmse_model, log_rmse_clim

Inits default to 00 UTC only (greenness is a daily state; four inits a day would quadruple
the rows with the same state); --init_stride keeps every n-th init day (7 = weekly, which
also thins the serial correlation between consecutive inits). Windows day/night, days 1 and 10.

    python scripts/evaluation/hydrostates_init_dataset.py --eval_name eval_mtg --init_stride 7
    python scripts/evaluation/hydrostates_init_dataset.py --eval_name train_mtg_new --init_stride 7
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from hydrostates_dataset import site_table  # noqa: E402
from hydrostates_runs import CONDITIONS, OUT_DIR, SEEDS, run_name  # noqa: E402

from deep_lst.utils.io import ExperimentPaths  # noqa: E402

PATHS = ExperimentPaths()
SITE_SETS = {"train_mtg_new": "training"}  # every other eval is a holdout
REFERENCES = ("clim", "persistence", "era5land")
STATES = {
    "relative_greenness": "greenness",
    "wetness": "wetness",
    "rzsm_wetness": "rzsm_wetness",
}  # by_cell coord -> column


def run_rows(eval_name, condition, seed, windows, days, init_hours, init_stride, sites):
    path = PATHS.results_dir(run_name(condition, seed), "version_0", eval_name) / "verif" / "by_cell.nc"
    if not path.exists():
        print(f"missing: {path}")
        return None
    cell = xr.open_dataset(path, decode_timedelta=False)
    inits = pd.DatetimeIndex(cell["init"].values)
    keep = np.isin(inits.hour, init_hours) & (((inits.normalize() - inits.normalize().min()).days % init_stride) == 0)
    cell = cell.isel(init=keep)
    n_sites, n_inits = cell.sizes["pixel"], cell.sizes["init"]
    frames = []
    for window in windows:
        for day in days:
            selected = cell.sel(window=window, day=day)
            rows = pd.DataFrame(np.repeat(sites.values, n_inits, axis=0), columns=sites.columns)
            rows["site_set"] = SITE_SETS.get(eval_name, "holdout")
            rows["condition"] = condition
            rows["seed"] = seed
            rows["run"] = f"{condition}_s{seed}"
            rows["init"] = np.tile(cell["init"].values, n_sites)
            rows["window"] = window
            rows["day"] = day
            for coord, column in STATES.items():
                if coord in cell.coords:
                    rows[column] = cell[coord].values.T.ravel()  # (init, pixel) -> pixel-major
            rows["coverage"] = cell["lookback_coverage"].values.T.ravel()
            rows["n_valid"] = selected["n"].values.T.ravel().astype(np.int64)
            rows["rmse_model"] = np.sqrt(selected["mse_model"].values.T.ravel())
            rows["bias_model"] = selected["bias_model"].values.T.ravel()
            for reference in REFERENCES:
                rows[f"rmse_{reference}"] = np.sqrt(selected[f"mse_{reference}"].values.T.ravel())
            rows["skill_clim"] = 1.0 - rows["rmse_model"] / rows["rmse_clim"]
            rows["log_rmse_model"] = np.log(rows["rmse_model"])
            rows["log_rmse_clim"] = np.log(rows["rmse_clim"])
            frames.append(rows[rows["n_valid"] > 0])
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--windows", nargs="+", default=["day", "night"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--init_hours", type=int, nargs="+", default=[0])
    parser.add_argument("--init_stride", type=int, default=1, help="keep every n-th init day")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR / "v02")
    args = parser.parse_args()

    sites = site_table(args.eval_name)
    frames = []
    for condition in args.conditions:
        for seed in SEEDS:
            rows = run_rows(
                args.eval_name, condition, seed, args.windows, args.days, args.init_hours, args.init_stride, sites
            )
            if rows is not None:
                frames.append(rows)
    dataset = pd.concat(frames, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"cells_{args.eval_name}.csv"
    dataset.to_csv(out_path, index=False, float_format="%.6g", date_format="%Y-%m-%dT%H")
    states = " | ".join(
        f"{column} valid {dataset[column].notna().mean():.1%}" for column in STATES.values() if column in dataset
    )
    print(
        f"{out_path}: {len(dataset)} rows | runs {sorted(dataset.run.unique())} | inits {dataset.init.nunique()} | "
        f"windows {sorted(dataset.window.unique())} | {states} | coverage valid {dataset.coverage.notna().mean():.1%}"
    )


if __name__ == "__main__":
    main()
