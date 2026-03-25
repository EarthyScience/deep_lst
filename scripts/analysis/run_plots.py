import argparse

import xarray as xr

from deep_lst.analysis.plots import PlotResults
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment name")
    parser.add_argument("--version", type=str, required=True, help="Experiment version")
    parser.add_argument("--variable", type=str, required=True, help="Variable to plot (e.g. LST)")
    parser.add_argument(
        "--init_time",
        type=int,
        default=0,
        help="Integer, index which inititial time to plot selected from available forecast.nc. Default to 0.",
    )
    return parser.parse_args()


def run_plots(results, variable, init_time=0):
    args = parse_args()

    plotter = PlotResults(results, variable, init_time, args.config, args.version)
    plotter.regression_parity()
    plotter.prediction_maps(lead_time=0)
    plotter.plume_diagram_lts(n_pixels=10)
    plotter.error_vs_leadtime(n_pixels=500)


if __name__ == "__main__":
    args = parse_args()

    results = xr.open_dataset(PATHS.predictions_path(args.config, args.version))
    run_plots(results, args.variable, args.init_time)
