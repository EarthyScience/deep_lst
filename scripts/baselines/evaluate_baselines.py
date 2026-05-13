"""Fast evaluation of LST baselines: ERA5 and SEVIRI climatologies.

Generates a fingerprint plot (DOY × hour heatmap of spatially-averaged LST)
for each available baseline in the specified eval region, in a single
multi-panel figure.

Baselines loaded (if available):
  - ERA5 skin temperature           assets/era5/baselines/baseline_{eval}.nc
  - SEVIRI hourly climatology       assets/climatology/hourly/climatology_{eval}.nc
  - SEVIRI smoothed climatology     assets/climatology/smoothed_hourly/climatology_{eval}.nc

Output
------
  results/baselines/{eval}/fingerprint.png

Usage
-----
python scripts/benchmarks/evaluate_baselines.py --eval dahra

# All regions:
for EVAL in dahra antananarivo gobabeb johannesburg east_africa sahel_niger w_africa; do
    python scripts/benchmarks/evaluate_baselines.py --eval ${EVAL}
done
"""

import argparse
from pathlib import Path

import xarray as xr

from deep_lst.evaluation.plots.fingerprint import fingerprint_plot
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval region name, e.g. dahra")
    p.add_argument(
        "--plots",
        nargs="+",
        default=["fingerprint"],
        choices=["fingerprint"],
        help="Plots to generate (default: fingerprint)",
    )
    return p.parse_args()


def _try_load(path: Path, label: str) -> xr.Dataset | None:
    if not path.exists():
        print(f"  [skip] {label}: file not found at {path}")
        return None
    ds = xr.open_dataset(path)
    print(f"  [ok]   {label}: {path.name}  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
    return ds


def evaluate_baselines(eval_name: str, plots: list[str]) -> None:
    print(f"\n=== Baseline evaluation: {eval_name} ===")

    # --- Load baselines ---
    datasets = {}

    ds_era5 = _try_load(PATHS.baseline_path(eval_name), "ERA5")
    if ds_era5 is not None:
        datasets["ERA5"] = ds_era5

    ds_hourly = _try_load(PATHS.climatology_path(eval_name, "hourly"), "SEVIRI (hourly)")
    if ds_hourly is not None:
        datasets["SEVIRI\n(hourly clim)"] = ds_hourly

    ds_smooth = _try_load(PATHS.climatology_path(eval_name, "smoothed_hourly"), "SEVIRI (smoothed)")
    if ds_smooth is not None:
        datasets["SEVIRI\n(smoothed clim)"] = ds_smooth

    if not datasets:
        print("No baseline files found. Run the extraction scripts first.")
        return

    out_dir = Path("results") / "baselines" / eval_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if "fingerprint" in plots:
        fingerprint_plot(
            datasets,
            title=f"LST fingerprint — {eval_name}",
            save_path=out_dir / "fingerprint.png",
        )

    print(f"Outputs written to: {out_dir}")


def main():
    args = parse_args()
    evaluate_baselines(args.eval, args.plots)


if __name__ == "__main__":
    main()
