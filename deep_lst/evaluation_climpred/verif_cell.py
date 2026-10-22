"""Per-cell verification: keeps init AND pixel, pools only the 24 leads of each forecast day.

Raw material for init-level models: mse of the model and of every reference forecast plus the
model's mean error (bias) per (window, init, pixel, day) with the model's valid sample counts,
carrying the (init, pixel) coordinates the state analysis needs: relative_greenness (vegetation
state at init), the hydrological states of assets/static/hydro_init_state_<selection>.nc when
present, and lookback_coverage (share of valid obs hours in the lookback). Squared errors are
taken on the same aligned cubes the other verifiers use for their counts (cubes.obs_cube /
baseline_cube). References are averaged over the model's samples (the paired convention of
common.build_ensembles), skipping the few cells where a reference itself is missing. The
solar-hour windows (masks.HOUR_BINS) resolve the diurnal cycle.

The window "amplitude" is not a mask but the diurnal amplitude error of each forecast day
(stats.amplitude_cells): model minus obs amplitude over the jointly observed hours, for days
with at least MIN_OBSERVED_HOURS such hours including one in the afternoon peak window and one
in the pre-dawn trough window (PEAK_HOURS, TROUGH_HOURS, local solar). It reuses the variable
names of the other windows (bias_model = signed amplitude error, mse_model, n = joint hours,
mse_<reference>) and adds amplitude_obs, amplitude_model_full and the two coverage flags.
Meaningful for 00 UTC inits, whose 24-lead day is a solar day; the init-level export keeps those.

Run: python -m deep_lst.evaluation_climpred.verif_cell --experiment <exp> --eval_name eval_mtg \\
         --windows all day night amplitude h00_04 h04_08 h08_12 h12_16 h16_20 h20_24
"""

import xarray as xr

from .common import baseline_files, setup_verif, verification_dir
from .cubes import baseline_cube, obs_cube
from .masks import MASKS, solar_hour_mask
from .stats import amplitude_cells

REFERENCES = {"clim_smooth": "clim", "persistence": "persistence", "era5land": "era5land"}  # file key -> mse_<name>
AMPLITUDE_WINDOW = "amplitude"
PEAK_HOURS = (12, 16)  # local solar hours [lo, hi) in which the afternoon maximum must be observed
TROUGH_HOURS = (3, 7)  # ... and the pre-dawn minimum


def main():
    args, initilised, obs = setup_verif()
    observed = obs_cube(obs, initilised)
    signed_error = initilised["LST"] - observed
    errors = {"model": signed_error**2}
    references = {}
    files = baseline_files(args.paths, args.spatial)
    for baseline, name in REFERENCES.items():
        reference = xr.DataArray(
            baseline_cube(files[baseline], initilised), dims=("init", "lead", "pixel"), coords=initilised.coords
        )
        references[name] = reference
        errors[name] = (reference - observed) ** 2
    valid = errors["model"].notnull()  # model and obs: the model's samples
    day = xr.DataArray((initilised["lead"].values - 1) // 24 + 1, dims="lead", name="day")

    per_window = []
    for window in args.windows:
        if window == AMPLITUDE_WINDOW:
            peak = solar_hour_mask(PEAK_HOURS)(initilised)
            trough = solar_hour_mask(TROUGH_HOURS)(initilised)
            per_window.append(amplitude_cells(initilised["LST"], observed, references, peak, trough))
            continue
        mask_fn = MASKS[window]
        keep = valid if mask_fn is None else valid & mask_fn(initilised)
        scores = {f"mse_{name}": error.where(keep).groupby(day).mean("lead") for name, error in errors.items()}
        scores["bias_model"] = signed_error.where(keep).groupby(day).mean("lead")
        scores["n"] = keep.groupby(day).sum("lead")
        per_window.append(xr.Dataset(scores))
    out = xr.concat(per_window, dim="window").assign_coords(window=args.windows)
    out.to_netcdf(verification_dir(args, "verif") / "by_cell.nc")


if __name__ == "__main__":
    main()
