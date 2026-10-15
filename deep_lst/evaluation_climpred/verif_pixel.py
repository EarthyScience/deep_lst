"""Per-pixel verification: keeps the pixel dim, reduces init (dim=["init"]).

Saves raw material only (mse + counts); dispersion, maps and forecast limits are
derived in derive.py. Run from notebooks/: python -m deep_lst.evaluation_climpred.verif_pixel
"""

import xarray as xr

from .common import VERIFY_KWARGS, build_ensembles, setup_verif, verification_dir
from .stats import per_pixel_counts


def main():
    args, initilised, obs = setup_verif()
    hindcast, baselines = build_ensembles(args, initilised, obs)

    model_px = hindcast.verify(dim=["init"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")
    baseline_px = baselines.verify(dim=["init"], **VERIFY_KWARGS)["LST"].drop_vars("skill", errors="ignore")
    # Squaring destroys the sign, so bias needs its own pass: climpred's me = f - o,
    # positive where the forecast runs warm. Pools over leads linearly by count.
    bias_kwargs = dict(VERIFY_KWARGS, metric="me")
    bias_px = hindcast.verify(dim=["init"], **bias_kwargs)["LST"].drop_vars("skill", errors="ignore")

    out = xr.Dataset(
        {
            "mse_model": model_px,
            "mse_baseline": baseline_px,
            "bias_model": bias_px,
            "n": per_pixel_counts(initilised, obs, args.windows),
        }
    )
    # A month-windows run gets its own file: it must not overwrite the by_pixel.nc
    # the maps and tables derive from.
    name = "by_pixel_monthly.nc" if args.monthly else "by_pixel.nc"
    out.to_netcdf(verification_dir(args, "verif") / name)


if __name__ == "__main__":
    main()
