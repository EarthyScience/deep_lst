"""ERA5 LST error vs surface drivers — multi-region spatial/seasonal GAM (bam), 2024.

Builds a tidy per-(region, pixel, month) table of daytime/nighttime ERA5 RMSE and candidate drivers
(FVC, albedo, DEM, diurnal amplitude), then fits mgcv bam() GAMs to ask which drivers structure the
error once season and smooth space are controlled, and to partition the explained deviance into
covariate-unique / space-unique / shared (confounded) parts.

Model (per response, vegetated pixels FVC>0, Gamma-log):
    rmse ~ region                                    # per-region intercepts
         + s(fvc)+s(albedo)+s(dem)+s(amplitude)
         + s(month, bs='cc', by=region)              # fixed per-region seasonal curve
         + ti(fvc, month, bs=c('tp','cc'))           # does the FVC effect change over the year
         + te(lon, lat, by=region)                   # per-region smooth space (lon/lat not comparable across regions)

Run in the R-enabled env (rpy2 needs R_HOME, which `mamba run` sets):
    mamba run -n deep_lst_rgam python deep_lst/exploration/residual_analysis.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))  # bam(discrete) uses these

ERA5_PATTERN = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/era5/eval_{}.nc"
SAMPLED_PATTERN = "/Net/Groups/BGI/tscratch/mwessel/wegen-stores/sampled_stores/seviri_sampled_eval_{}.zarr"
REGIONS = [
    "antananarivo",
    "bouamir",
    "dahra",
    "gobabeb",
    "johannesburg",
    "passage",
    "sahel_niger",
    "west_africa",
    "yangambi",
]
RUN_NAME = "era5_allregions"  # output subfolder under results/gam
OUT_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/gam") / RUN_NAME
YEAR = 2024
DAY_HOURS = (10, 15)  # inclusive UTC valid-hour window
NIGHT_HOURS = (22, 3)  # wraps midnight


def region_table(region):
    """Tidy per-(pixel, month) table for one region: day/night ERA5 RMSE + drivers, for YEAR."""
    era5 = xr.open_dataset(ERA5_PATTERN.format(region))
    sampled = xr.open_zarr(SAMPLED_PATTERN.format(region))
    era5_time = pd.DatetimeIndex(era5["valid_time"].values)
    sampled_time = pd.DatetimeIndex(sampled["time_idx"].values)

    grid_lat, grid_lon = era5["lat"].values, era5["lon"].values
    pix_lat, pix_lon = sampled["lat"].values, sampled["lon"].values
    nearest_lat = np.abs(pix_lat[:, None] - grid_lat[None, :]).argmin(axis=1)
    nearest_lon = np.abs(pix_lon[:, None] - grid_lon[None, :]).argmin(axis=1)
    dem = sampled["DEM"].values
    landcov = sampled["LANDCOV"].values
    n_pixel = sampled.sizes["spatial_idx"]

    months = []
    for month in range(1, 13):
        era5_idx = np.where((era5_time.year == YEAR) & (era5_time.month == month))[0]
        sampled_idx = np.where((sampled_time.year == YEAR) & (sampled_time.month == month))[0]
        if len(era5_idx) == 0 or len(era5_idx) != len(sampled_idx) or len(era5_idx) % 24 != 0:
            continue

        era5_pix = era5["skt"].isel(valid_time=era5_idx).values[:, nearest_lat, nearest_lon]
        lst = sampled["LST"].isel(time_idx=sampled_idx).transpose("time_idx", "spatial_idx").values
        fvc = sampled["FVC"].isel(time_idx=sampled_idx).transpose("time_idx", "spatial_idx").values
        albedo = sampled["AL-BB-DH"].isel(time_idx=sampled_idx).transpose("time_idx", "spatial_idx").values
        hour = pd.DatetimeIndex(era5["valid_time"].values[era5_idx]).hour.values

        squared_error = (era5_pix - lst) ** 2
        day_mask = (hour >= DAY_HOURS[0]) & (hour <= DAY_HOURS[1])
        night_mask = (hour >= NIGHT_HOURS[0]) | (hour <= NIGHT_HOURS[1])
        n_day = len(era5_idx) // 24
        with np.errstate(invalid="ignore"):
            rmse_day = np.sqrt(np.nanmean(squared_error[day_mask], axis=0))
            rmse_night = np.sqrt(np.nanmean(squared_error[night_mask], axis=0))
            mean_cycle = np.nanmean(lst.reshape(n_day, 24, -1), axis=0)
            amplitude = np.nanpercentile(mean_cycle, 95, axis=0) - np.nanpercentile(mean_cycle, 5, axis=0)
            fvc_pixel = np.nanmean(fvc, axis=0)
            albedo_pixel = np.nanmean(albedo, axis=0)

        months.append(
            pd.DataFrame(
                {
                    "region": region,
                    "pixel": np.arange(n_pixel),
                    "month": month,
                    "rmse_day": rmse_day,
                    "rmse_night": rmse_night,
                    "fvc": fvc_pixel,
                    "albedo": albedo_pixel,
                    "amplitude": amplitude,
                    "dem": dem,
                    "lat": pix_lat,
                    "lon": pix_lon,
                    "landcov": landcov,
                }
            )
        )

    return pd.concat(months, ignore_index=True)


def build_table():
    """Concatenate per-region tables into one multi-region (region, pixel, month) table."""
    tables = []
    for region in REGIONS:
        tables.append(region_table(region))
        print(f"  built {region}: {len(tables[-1])} rows")
    table = pd.concat(tables, ignore_index=True)
    table["is_bare"] = (table["fvc"] == 0).astype(int)
    return table


# --- GAM (mgcv via rpy2) -----------------------------------------------------
import rpy2.robjects as ro  # noqa: E402
from rpy2.robjects import pandas2ri  # noqa: E402
from rpy2.robjects.conversion import localconverter  # noqa: E402
from rpy2.robjects.packages import importr  # noqa: E402

mgcv = importr("mgcv")
grdevices = importr("grDevices")
GAMMA_LOG = ro.r('Gamma(link="log")')
MONTH_KNOTS = ro.r("list(month = c(0.5, 12.5))")  # cyclic wrap between Dec and Jan

_SMOOTHS = "s(fvc) + s(albedo) + s(dem) + s(amplitude)"
_SEASON = "s(month, bs='cc', k=6, by=region)"
_FVC_SEASON = "ti(fvc, month, bs=c('tp','cc'), k=c(6, 6))"
_SPACE = "te(lon, lat, by=region)"

FORMULA_FULL = f"{{y}} ~ region + {_SMOOTHS} + {_SEASON} + {_FVC_SEASON} + {_SPACE}"
FORMULA_COV = f"{{y}} ~ region + {_SMOOTHS} + {_SEASON} + {_FVC_SEASON}"
FORMULA_SPACE = "{y} ~ region + " + _SPACE

GAM_COLUMNS = ["fvc", "albedo", "dem", "amplitude", "month", "lon", "lat", "region"]


def _fit(formula, r_df):
    return mgcv.bam(
        ro.Formula(formula),
        data=r_df,
        family=GAMMA_LOG,
        knots=MONTH_KNOTS,
        method="fREML",
        discrete=True,
        nthreads=N_THREADS,
    )


def _dev_expl(fit):
    return float(ro.r["summary"](fit).rx2("dev.expl")[0])


def _summary_text(fit):
    return "\n".join(ro.r["capture.output"](ro.r["summary"](fit)))


def _smooth_table(fit, response):
    s_table = ro.r["summary"](fit).rx2("s.table")
    df = pd.DataFrame(np.array(s_table), columns=list(ro.r["colnames"](s_table)))
    df.insert(0, "term", list(ro.r["rownames"](s_table)))
    df.insert(0, "response", response)
    return df


def _save_concurvity(fit, response, out_dir):
    conc = mgcv.concurvity(fit, full=True)
    df = pd.DataFrame(np.array(conc), index=list(ro.r["rownames"](conc)), columns=list(ro.r["colnames"](conc)))
    df.to_csv(out_dir / f"concurvity_{response}.csv")


def _save_partial_effects(fit, veg, response, out_dir):
    grdevices.pdf(str(out_dir / f"partial_effects_{response}.pdf"), width=12, height=8)
    ro.r["plot"](fit, pages=0, scheme=2, shade=True, seWithMean=True)
    grdevices.dev_off()

    terms = ro.r["predict"](fit, type="terms")
    effects = pd.DataFrame(np.array(terms), columns=[f"effect:{c}" for c in ro.r["colnames"](terms)])
    out = pd.concat([veg[GAM_COLUMNS].reset_index(drop=True), effects], axis=1)
    out.to_csv(out_dir / f"partial_effects_{response}.csv", index=False)


def fit_gam(table, response, tod_label, out_dir):
    """Gamma-log bam() on vegetated pixels; variance-partition + save summary/rds/partial/concurvity."""
    keep = ["fvc", "albedo", "dem", "amplitude", response]
    veg = table[(table["fvc"] > 0) & (table[response] > 0)].dropna(subset=keep + ["lon", "lat"]).copy()
    veg["region"] = veg["region"].astype("category")
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(veg[GAM_COLUMNS + [response]])

    full = _fit(FORMULA_FULL.format(y=response), r_df)
    cov = _fit(FORMULA_COV.format(y=response), r_df)
    space = _fit(FORMULA_SPACE.format(y=response), r_df)
    d_full, d_cov, d_space = _dev_expl(full), _dev_expl(cov), _dev_expl(space)

    print(f"\n===== {tod_label}: vegetated pixels (n={len(veg)}), Gamma(log) bam =====")
    print(f"deviance explained:  full={d_full:.3f}  covariates-only={d_cov:.3f}  space-only={d_space:.3f}")
    print(f"  covariate-unique a = full - space      = {d_full - d_space:.3f}")
    print(f"  space-unique     b = full - covariates = {d_full - d_cov:.3f}")
    print(f"  shared/confounded c = cov + space - full = {d_cov + d_space - d_full:.3f}")
    print(ro.r["summary"](full))

    (out_dir / f"gam_summary_{response}.txt").write_text(_summary_text(full))
    ro.r["saveRDS"](full, file=str(out_dir / f"gam_{response}.rds"))
    _save_concurvity(full, response, out_dir)
    _save_partial_effects(full, veg, response, out_dir)

    partition = {
        "response": response,
        "n": len(veg),
        "dev_full": d_full,
        "dev_cov": d_cov,
        "dev_space": d_space,
        "a_cov_unique": d_full - d_space,
        "b_space_unique": d_full - d_cov,
        "c_shared": d_cov + d_space - d_full,
    }
    return partition, _smooth_table(full, response)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    print("table rows:", len(table), "| regions:", table["region"].nunique())

    partitions, smooth_tables = [], []
    for response, tod_label in (("rmse_day", "DAYTIME"), ("rmse_night", "NIGHTTIME")):
        partition, smooth = fit_gam(table, response, tod_label, OUT_DIR)
        partitions.append(partition)
        smooth_tables.append(smooth)

    pd.DataFrame(partitions).to_csv(OUT_DIR / "partition.csv", index=False)
    pd.concat(smooth_tables, ignore_index=True).to_csv(OUT_DIR / "terms.csv", index=False)
    print("\nresults saved to", OUT_DIR)
