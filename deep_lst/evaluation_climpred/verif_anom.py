"""Anomaly verification: centered ACC (pearson_r), the obs-anomaly variance for the
hourly anomaly R2, and the anomaly moment sums from which any correlation, variance
or mse pooled over leads (the day-level table cells) follows exactly.
Run: python -m deep_lst.evaluation_climpred.verif_anom
"""

import xarray as xr

from .common import VERIFY_KWARGS, baseline_files, build_anomaly_ensemble, setup_verif, verification_dir
from .cubes import baseline_cube
from .stats import anomaly_moments, anomaly_obs_variance


def main():
    args, initilised, obs = setup_verif()
    acc_hindcast, anomaly_obs = build_anomaly_ensemble(args, initilised, obs)

    acc = acc_hindcast.verify(dim=["init", "pixel"], **{**VERIFY_KWARGS, "metric": "pearson_r"})
    acc = acc["LST"].drop_vars("skill", errors="ignore")

    # The forecast anomaly is only returned stacked by build_anomaly_ensemble; the
    # moment sums need it unstacked with the init/lead/pixel coords the masks read.
    clim_path = baseline_files(args.paths, args.spatial)["clim_smooth"]
    anomaly_forecast = initilised["LST"] - baseline_cube(clim_path, initilised)
    moments = anomaly_moments(initilised, anomaly_forecast, anomaly_obs, args.windows)

    out = xr.Dataset(
        {
            "acc_centered": acc,
            "anomaly_obs_variance": anomaly_obs_variance(initilised, anomaly_obs, args.windows),
        }
    )
    out = out.merge(moments)
    out.to_netcdf(verification_dir(args, "verif") / "anom.nc")


if __name__ == "__main__":
    main()
