"""Cross-run table of the input sensitivity: per variable, window and forecast day the RMSE
increase of each run's summary.csv, averaged over the runs with its range.

    python scripts/sensitivity/summarise_runs.py --experiments mlp_smoothl1_h24_full_v16_hydrostates \\
        mlp_smoothl1_h24_full_v16_hydrostates_s7 mlp_smoothl1_h24_full_v16_hydrostates_s23 \\
        --eval_name eval_mtg --out results/independent/sensitivity/full_v16_hydrostates_eval_mtg.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from deep_lst.sensitivity.forecast import sensitivity_dir
from deep_lst.utils.io import ExperimentPaths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--eval_name", default="eval_mtg")
    parser.add_argument("--version", default="version_0")
    parser.add_argument("--method", default="permute_days")
    parser.add_argument("--block", default="month")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = ExperimentPaths()
    frames = []
    for experiment in args.experiments:
        folder = sensitivity_dir(paths, experiment, args.version, args.eval_name, args.method, args.block)
        table = pd.read_csv(folder / "summary.csv")
        table["experiment"] = experiment
        frames.append(table)
    runs = pd.concat(frames, ignore_index=True)
    runs["delta_pct"] = 100 * runs["delta_rmse"] / runs["rmse_reference"]

    summary = (
        runs.groupby(["day", "window", "variable"])
        .agg(
            rmse_reference=("rmse_reference", "mean"),
            delta_rmse=("delta_rmse", "mean"),
            delta_rmse_min=("delta_rmse", "min"),
            delta_rmse_max=("delta_rmse", "max"),
            delta_pct=("delta_pct", "mean"),
            share_pixels_worse=("share_pixels_worse", "mean"),
            n_runs=("experiment", "nunique"),
        )
        .reset_index()
        .sort_values(["day", "window", "delta_rmse"], ascending=[True, True, False])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False, float_format="%.4f")
    print(args.out)
    print(summary[(summary["window"] == "all") & (summary["day"] == 1)].to_string(index=False))


if __name__ == "__main__":
    main()
