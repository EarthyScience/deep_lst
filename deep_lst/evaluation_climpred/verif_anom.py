"""Anomaly verification: centered ACC (pearson_r) plus the obs-anomaly variance
moments for the anomaly R2. Run: python -m deep_lst.evaluation_climpred.verif_anom
"""

import xarray as xr

from .common import VERIFY_KWARGS, build_anomaly_ensemble, setup_verif, verification_dir
from .stats import anomaly_obs_variance


def main():
    args, initilised, obs = setup_verif()
    acc_hindcast, anomaly_obs = build_anomaly_ensemble(args, initilised, obs)

    acc = acc_hindcast.verify(dim=["init", "pixel"], **{**VERIFY_KWARGS, "metric": "pearson_r"})
    acc = acc["LST"].drop_vars("skill", errors="ignore")

    out = xr.Dataset(
        {
            "acc_centered": acc,
            "anomaly_obs_variance": anomaly_obs_variance(initilised, anomaly_obs, args.windows),
        }
    )
    out.to_netcdf(verification_dir(args, "verif") / "anom.nc")


if __name__ == "__main__":
    main()
