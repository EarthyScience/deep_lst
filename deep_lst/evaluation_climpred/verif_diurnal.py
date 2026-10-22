"""Diurnal verification: the error resolved by local solar hour and forecast day, and the
diurnal amplitude error by forecast day -- how the daily dynamics get harder with lead.

No climpred. Per (solar hour, forecast day, pixel) the sums of the model's squared and
signed errors, the obs sums (NSE) and the anomaly sums (ACC) over the samples where
model, obs and climatology are all valid, and the error and obs sums of every reference
forecast (climatology, persistence, ERA5-Land SKT: verif_cell.REFERENCES) on those
samples where the reference is itself valid (stats.diurnal_sums); per (forecast day,
pixel) the sums of the amplitude errors of the model and of the references over the
(init, day) pairs with at least MIN_OBSERVED_HOURS valid hours (stats.amplitude_sums).
derive.diurnal_scores pools the pixels into the regimes; evaluate draws the diurnal
figures when verif/diurnal.nc exists. --windows is accepted but unused: the solar hour
replaces the windows.

Run: python -m deep_lst.evaluation_climpred.verif_diurnal --experiment <exp> --eval_name <name>
"""

import xarray as xr

from .common import baseline_files, setup_verif, verification_dir
from .cubes import baseline_cube, obs_cube
from .stats import amplitude_sums, diurnal_sums
from .verif_cell import REFERENCES

PIXEL_COORDS = ("lat", "lon", "kg_class", "lc_class")


def main():
    args, initilised, obs = setup_verif()
    observed = obs_cube(obs, initilised).values
    files = baseline_files(args.paths, args.spatial)
    references = {}
    for key, name in REFERENCES.items():
        references[name] = baseline_cube(files[key], initilised)

    hourly = diurnal_sums(initilised, observed, references)
    amplitude = amplitude_sums(initilised, observed, references)
    amplitude = amplitude.rename({name: f"amplitude_{name}" for name in amplitude.data_vars})
    out = xr.merge([hourly, amplitude])
    for coord in PIXEL_COORDS:
        out = out.assign_coords({coord: ("pixel", initilised[coord].values)})
    out.to_netcdf(verification_dir(args, "verif") / "diurnal.nc")


if __name__ == "__main__":
    main()
