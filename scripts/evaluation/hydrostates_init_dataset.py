"""Init-level dataset of the hydrostates runs for the mixed models (lme4): one CSV per eval.

One row per site x init x solar window x forecast day x run, from verif/by_cell.nc
(verif_cell.py):

    site, lon, lat, in_training, kg_class, kg_major, lc_class, ecoregion, biome, aridity, elevation,
    fvc_annual_mean, fvc_class, lst_validity     site attributes (aridity = Global Aridity Index v3.1)
    site_set         holdout | training (train_* evals), for stacking the two files
    condition, seed, run, init, window, lead     window = solar window (day/night), solar-hour bin (h00_04 ...)
                                                 or "amplitude" (see below); lead = forecast day (1-10), the
                                                 24 hourly leads of that day pooled
    greenness        FVC on the init day relative to the site's climatological annual range (0-1)
    wetness          ERA5-Land swvl1 on the init day relative to the site's climatological annual range (0-1)
    wetness_lookback, wetness_tendency   mean over the 7 days ending at the init / init minus 7 days earlier
    rzsm_wetness, rzsm_wetness_lookback, rzsm_wetness_tendency   the same for the GloFAS root-zone soil
                     wetness index (where the asset has it)
    coverage         share of valid LST hours in the lookback window before the init (0-1)
    n_valid          jointly valid (forecast, obs) hours pooled into the row
    rmse_model, bias_model, rmse_clim, rmse_persistence, rmse_era5land   over the model's samples
    skill_clim, log_rmse_model, log_rmse_clim, log_rmse_persistence
    amplitude rows (window == "amplitude", verif_cell / stats.amplitude_cells): bias_model = model minus obs
    diurnal amplitude over the jointly observed hours of the forecast day (K), rmse_model = |bias_model|,
    rmse_<reference> likewise for the references, n_valid = joint hours, and
    amplitude_obs            observed amplitude over those hours (K)
    amplitude_model_full     model amplitude over all 24 hours (K); minus (amplitude_obs + bias_model) = truncation
    amplitude_ratio          (amplitude_obs + bias_model) / amplitude_obs, 1 = the forecast reproduces the range
    peak_observed, trough_observed   coverage flags of the rule (>= 12 joint hours, solar 12-16 h and 3-7 h seen)

Inits default to 00 UTC only (greenness is a daily state; four inits a day would quadruple
the rows with the same state); --init_stride keeps every n-th init day (7 = weekly, which
also thins the serial correlation between consecutive inits). Windows day/night, days 1 and 10.

    python scripts/evaluation/hydrostates_init_dataset.py --eval_name eval_mtg --init_stride 7
    python scripts/evaluation/hydrostates_init_dataset.py --eval_name train_mtg_new --init_stride 7

--version selects the run matrix (hydrostates_runs.VERSIONS); v03 = the v16.2 matrix, all ten days;
v04 = v03 plus the amplitude rows and the lookback wetness:

    python scripts/evaluation/hydrostates_init_dataset.py --version v16.2 --eval_name eval_mtg_hydrostates_val \
        --windows day night amplitude --init_stride 7 --days 1 2 3 4 5 6 7 8 9 10 \
        --out_dir results/independent/hydrostates/v04

Rows are appended to the CSV run by run, so the training file never sits in memory whole.
(v01/v02 files call the lead column `day`.)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from hydrostates_dataset import site_table  # noqa: E402
from hydrostates_runs import CONDITIONS, OUT_DIR, VERSIONS, run_name  # noqa: E402

from deep_lst.utils.io import ExperimentPaths  # noqa: E402

PATHS = ExperimentPaths()
REFERENCES = ("clim", "persistence", "era5land")
STATES = {
    "relative_greenness": "greenness",
    "wetness": "wetness",
    "wetness_lookback": "wetness_lookback",
    "wetness_tendency": "wetness_tendency",
    "rzsm_wetness": "rzsm_wetness",
    "rzsm_wetness_lookback": "rzsm_wetness_lookback",
    "rzsm_wetness_tendency": "rzsm_wetness_tendency",
}  # by_cell coord -> column
AMPLITUDE_COLUMNS = ("amplitude_obs", "amplitude_model_full", "peak_observed", "trough_observed")  # amplitude window


def run_rows(eval_name, condition, seed, version, windows, days, init_hours, init_stride, sites):
    path = PATHS.results_dir(run_name(condition, seed, version), "version_0", eval_name) / "verif" / "by_cell.nc"
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
            rows["site_set"] = "training" if eval_name.startswith("train_") else "holdout"
            rows["condition"] = condition
            rows["seed"] = seed
            rows["run"] = f"{condition}_s{seed}"
            rows["init"] = np.tile(cell["init"].values, n_sites)
            rows["window"] = window
            rows["lead"] = day
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
            rows["log_rmse_persistence"] = np.log(rows["rmse_persistence"])
            for column in AMPLITUDE_COLUMNS:
                if column in selected:
                    rows[column] = selected[column].values.T.ravel()
            if window == "amplitude":
                rows["amplitude_ratio"] = (rows["amplitude_obs"] + rows["bias_model"]) / rows["amplitude_obs"]
            frames.append(rows[rows["n_valid"] > 0])
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--version", default="v16", choices=list(VERSIONS), help="run matrix")
    parser.add_argument("--conditions", nargs="+", default=None, choices=list(CONDITIONS), help="default: the matrix's")
    parser.add_argument("--windows", nargs="+", default=["day", "night"])
    parser.add_argument("--days", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--init_hours", type=int, nargs="+", default=[0])
    parser.add_argument("--init_stride", type=int, default=1, help="keep every n-th init day")
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR / "v02")
    args = parser.parse_args()

    matrix = VERSIONS[args.version]
    conditions = args.conditions if args.conditions is not None else list(matrix["conditions"])
    sites = site_table(args.eval_name, training_selection=matrix["training"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"cells_{args.eval_name}.csv"
    n_rows = 0
    runs = []
    for condition in conditions:
        for seed in matrix["seeds"]:
            rows = run_rows(
                args.eval_name,
                condition,
                seed,
                args.version,
                args.windows,
                args.days,
                args.init_hours,
                args.init_stride,
                sites,
            )
            if rows is None:
                continue
            first = n_rows == 0
            rows.to_csv(
                out_path,
                mode="w" if first else "a",
                header=first,
                index=False,
                float_format="%.6g",
                date_format="%Y-%m-%dT%H",
            )
            n_rows += len(rows)
            runs.append(rows["run"].iloc[0])
            states = " | ".join(
                f"{column} valid {rows[column].notna().mean():.1%}" for column in STATES.values() if column in rows
            )
            print(
                f"  {runs[-1]}: {len(rows)} rows | inits {rows.init.nunique()} | {states} | "
                f"coverage valid {rows.coverage.notna().mean():.1%}"
            )
    print(f"{out_path}: {n_rows} rows | runs {runs}")


if __name__ == "__main__":
    main()
