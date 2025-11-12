import xarray as xr
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_lst.misc.utils import open_config
from deep_lst.misc.metrics import *
from deep_lst.misc.proc_results import *

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment", default = "", type=str, help="Name of the experiment.")
args = argparser.parse_args()   


def metrics_timeseries(tsteps, inference_results, df, land_mask, vars = ["targ", "pred"], metric = "mae"):

    ts = []
    if metric == "mae":
        score = mae
    elif metric == "rmse":
        score = rmse
    elif metric == "bias":
        score = bias
    elif metric == "absolute_error":
        score = absolute_error
    elif metric == "rsquared":
        score = r_squared
    elif metric == "anomaly_correlation":
        score = anomaly_correlation
    else:
        raise ValueError(f"Metric {metric} not recognized.")

    for i in range(tsteps):

        df.loc[land_mask == True, "targ"] = inference_results.loc[inference_results["timestep"] ==i]["targets"].values
        df.loc[land_mask == True, "pred"] = inference_results.loc[inference_results["timestep"] ==i]["predictions"].values

        ds = xr.Dataset({value: assemble_dataarray(df, value) for value in vars})

        mae_value = score(ds.pred, ds.targ, mean=True).item()
        ts.append(mae_value)
    return ts

def main(config, log_path):

    seviri_indices = pd.read_parquet(config["experiment_path"] + config["seviri"]["spatial_indices"])
    inference_results = pd.read_parquet(log_path + "inference_results.parquet")

    print(config['test_start'], config['test_end'])
    TSTEPS = len(pd.date_range(start=config['test_start'], end=config['test_end'], freq='1H'))-1
    print(TSTEPS)

    inference_results_r = reshape_results(inference_results, TSTEPS)
    df, land_mask = results_store(config, seviri_indices)
    mae_ts = metrics_timeseries(TSTEPS, inference_results_r, df, land_mask, vars = ["targ", "pred"], metric = "mae")
    rmse_ts = metrics_timeseries(TSTEPS, inference_results_r, df, land_mask, vars = ["targ", "pred"], metric = "rmse")
    bias_ts = metrics_timeseries(TSTEPS, inference_results_r, df, land_mask, vars = ["targ", "pred"], metric = "bias")
    rsquared_ts = metrics_timeseries(TSTEPS, inference_results_r, df, land_mask, vars = ["targ", "pred"], metric = "rsquared")
    anomaly_correlation_ts = metrics_timeseries(TSTEPS, inference_results_r, df, land_mask, vars = ["targ", "pred"], metric = "anomaly_correlation")

    df = pd.DataFrame({"MAE": mae_ts, "RMSE": rmse_ts, "Bias": bias_ts, "R_squared": rsquared_ts, "Anomaly_Correlation": anomaly_correlation_ts, "time_step": np.arange(TSTEPS)})
    df.to_parquet(log_path + "scoring_results.parquet")

if __name__ == "__main__":

    experiment = args.experiment

    config = open_config(experiment=experiment, path="./experiments")
    log_path = config['experiment_path'] + f"logs/prototype_{config['experiment_name']}/version_0/"

    main(config, log_path)

    print("Verification done.")
