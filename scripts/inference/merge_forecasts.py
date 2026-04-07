"""Merge partial forecast chunks from parallel inference runs into a single NetCDF."""

import argparse

import xarray as xr

from deep_lst.utils import ExperimentPaths


def parse_args():
    parser = argparse.ArgumentParser(description="Merge forecast chunk files into one")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--version", type=str, required=True, help="Version folder")
    parser.add_argument("--eval", type=str, required=True, help="Eval site name (without eval_ prefix)")
    parser.add_argument("--delete_chunks", action="store_true", help="Delete chunk files after merging")
    return parser.parse_args()


def merge_forecasts():
    args = parse_args()
    paths = ExperimentPaths()

    results_dir = paths.results_dir(args.experiment, args.version, args.eval)
    chunk_files = sorted(results_dir.glob("forecast_chunk*.nc"))

    if not chunk_files:
        print(f"No chunk files found in {results_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files:")
    for f in chunk_files:
        print(f"  - {f.name}")

    datasets = [xr.open_dataset(f) for f in chunk_files]
    merged = xr.concat(datasets, dim="init_time").sortby("init_time")

    output_path = paths.forecast_path(args.experiment, args.version, args.eval)
    merged.to_netcdf(output_path)
    print(f"Saved merged forecast to {output_path}")

    for ds in datasets:
        ds.close()

    if args.delete_chunks:
        for f in chunk_files:
            f.unlink()
            print(f"Deleted {f.name}")


if __name__ == "__main__":
    merge_forecasts()
