"""Per-cell verification: keeps init AND pixel, pools only the 24 leads of each forecast day.

Raw material for init-level models: mse of the model and of the smoothed climatology per
(window, init, pixel, day) with the jointly valid sample counts, carrying the (init, pixel)
coordinates the state analysis needs, relative_greenness (vegetation state at init) and
lookback_coverage (share of valid obs hours in the lookback). Squared errors are taken on the
same aligned cubes the other verifiers use for their counts (cubes.obs_cube / baseline_cube),
so a count-weighted mean over init reproduces by_pixel.nc exactly.

Run: python -m deep_lst.evaluation_climpred.verif_cell --experiment <exp> --eval_name eval_mtg
"""

import xarray as xr

from .common import baseline_files, setup_verif, verification_dir
from .cubes import baseline_cube, obs_cube
from .masks import MASKS


def main():
    args, initilised, obs = setup_verif()
    observed = obs_cube(obs, initilised)
    climatology = xr.DataArray(
        baseline_cube(baseline_files(args.paths, args.spatial)["clim_smooth"], initilised),
        dims=("init", "lead", "pixel"),
        coords=initilised.coords,
    )
    error_model = (initilised["LST"] - observed) ** 2
    error_climatology = (climatology - observed) ** 2
    valid = error_model.notnull() & error_climatology.notnull()
    day = xr.DataArray((initilised["lead"].values - 1) // 24 + 1, dims="lead", name="day")

    per_window = []
    for window in args.windows:
        mask_fn = MASKS[window]
        keep = valid if mask_fn is None else valid & mask_fn(initilised)
        n = keep.groupby(day).sum("lead")
        per_window.append(
            xr.Dataset(
                {
                    "mse_model": error_model.where(keep).groupby(day).sum("lead") / n,
                    "mse_clim": error_climatology.where(keep).groupby(day).sum("lead") / n,
                    "n": n,
                }
            )
        )
    out = xr.concat(per_window, dim="window").assign_coords(window=args.windows)
    out.to_netcdf(verification_dir(args, "verif") / "by_cell.nc")


if __name__ == "__main__":
    main()
