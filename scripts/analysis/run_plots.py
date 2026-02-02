import argparse

import xarray as xr

from deep_lst.analysis.plots import PlotResults
from deep_lst.utils import ExperimentPaths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment name")
    parser.add_argument("--version", type=str, required=True, help="Experiment version")
    return parser.parse_args()


PATHS = ExperimentPaths()

if __name__ == "__main__":
    args = parse_args()

    results = xr.open_dataset(PATHS.predictions_path(args.config, args.version))

    plotter = PlotResults(results, args.config, args.version)
    plotter.regression_parity()
    plotter.prediction_maps(lead_time=0)
    plotter.plume_diagram_lts(n_pixels=10)
