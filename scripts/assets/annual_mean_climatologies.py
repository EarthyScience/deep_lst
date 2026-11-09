"""Annual means of the monthly FVC and LST observational-validity climatologies.

Reads the twelve monthly NetCDFs of each climatology (work_5 derived, 2016-2024, full
seviri grid) and writes one annual file next to them, in the same layout:

    FVC_monthly_climatology/fvc_climatology_annual.nc                  fvc_mean, n_months
    LSTvalidity_monthly_climatology/lst_validity_climatology_annual.nc lst_validity, n_months

The annual mean is the mean over the months that have a value (NaN where no month has
one), i.e. the convention of fvc_site_classes.py; n_months counts the months that went in.
These files are the FVC and validity rasters of the site selection in eodata_slim
(scripts/site_selection/s2_majortom/common.py).

    python scripts/assets/annual_mean_climatologies.py
"""

from pathlib import Path

import numpy as np
import xarray as xr

FVC_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/FVC_monthly_climatology")
VALIDITY_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/LSTvalidity_monthly_climatology")


def annual_mean(directory, file_stem, variable):
    """Dataset with the annual mean of `variable` and the number of contributing months."""
    monthly = []
    for month in range(1, 13):
        field = xr.open_dataset(directory / f"{file_stem}_{month:02d}.nc")[variable]
        monthly.append(field.load())
    stacked = xr.concat(monthly, dim="month")

    mean = stacked.mean(dim="month", skipna=True).astype(np.float32)
    n_months = stacked.notnull().sum(dim="month").astype(np.int8)

    template = xr.open_dataset(directory / f"{file_stem}_01.nc")
    annual = xr.Dataset({variable: mean, "n_months": n_months})
    annual.attrs = dict(template.attrs)
    annual.attrs["month"] = "annual: mean over the months with a value, 1-12"
    annual[variable].attrs = dict(template[variable].attrs)
    annual["n_months"].attrs = {"long_name": "number of monthly climatologies with a value"}
    return annual


def main():
    for directory, file_stem, variable in [
        (FVC_DIR, "fvc_climatology", "fvc_mean"),
        (VALIDITY_DIR, "lst_validity_climatology", "lst_validity"),
    ]:
        annual = annual_mean(directory, file_stem, variable)
        out_path = directory / f"{file_stem}_annual.nc"
        annual.to_netcdf(out_path)
        n_valid = int(annual[variable].notnull().sum())
        print(f"wrote {out_path}: {n_valid} pixels with a value")


if __name__ == "__main__":
    main()
