"""Per-pixel verification: keeps the pixel dim, reduces init (dim=["init"]).

Saves raw material only: mse and bias of the model and of the baselines, the counts,
and the per-pixel sufficient statistics of the obs (NSE denominator, nRMSE span) and of
the anomalies (centered ACC, anomaly R2); dispersion, maps and forecast limits are
derived in derive.py. Run from notebooks/: python -m deep_lst.evaluation_climpred.verif_pixel
"""

import xarray as xr

from .common import VERIFY_KWARGS, baseline_files, build_ensembles, setup_verif, verification_dir
from .cubes import anomaly_pair
from .stats import anomaly_moments, per_pixel_obs_moments

ANOMALY_SUMS = ["n", "sum_f", "sum_o", "sum_ff", "sum_oo", "sum_fo"]  # stored as anom_<name>


def main():
    args, initilised, obs = setup_verif()
    hindcast, baselines = build_ensembles(args, initilised, obs)

    model_px = hindcast.verify(dim=["init"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")
    baseline_px = baselines.verify(dim=["init"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")
    # Squaring destroys the sign, so bias needs its own pass: climpred's me = f - o,
    # positive where the forecast runs warm. Pools over leads linearly by count. The
    # baselines' bias too: the reference scatter needs the climatology's sign per pixel.
    bias_kwargs = dict(VERIFY_KWARGS, metric="me")
    bias_px = hindcast.verify(dim=["init"], **bias_kwargs)["LST"].drop_vars("skill", errors="ignore")
    bias_baseline_px = baselines.verify(dim=["init"], **bias_kwargs)["LST"].drop_vars("skill", errors="ignore")
    del hindcast, baselines  # the stacked window copies: free them before the cubes below

    # Obs sums per pixel (NSE denominator, nRMSE span) and anomaly sums per pixel (ACC,
    # anomaly R2): the per-pixel counterparts of verif_anom's domain-wide moments.
    obs_moments = per_pixel_obs_moments(initilised, obs, args.windows)
    clim_path = baseline_files(args.paths, args.spatial)["clim_smooth"]
    anomaly_forecast, anomaly_obs = anomaly_pair(initilised, obs, clim_path)
    anomaly = anomaly_moments(initilised, anomaly_forecast, anomaly_obs, args.windows, dims=["init"])

    out = xr.Dataset(
        {
            "mse_model": model_px,
            "mse_baseline": baseline_px,
            "bias_model": bias_px,
            "bias_baseline": bias_baseline_px,
            "n": obs_moments["n"],
            "sum_o": obs_moments["sum_o"],
            "sum_oo": obs_moments["sum_oo"],
            "obs_range": obs_moments["obs_range"],
        }
    )
    for name in ANOMALY_SUMS:
        out[f"anom_{name}"] = anomaly[name]
    # A month-windows run gets its own file: it must not overwrite the by_pixel.nc
    # the maps and tables derive from.
    name = "by_pixel_monthly.nc" if args.monthly else "by_pixel.nc"
    out.to_netcdf(verification_dir(args, "verif") / name)


if __name__ == "__main__":
    main()
