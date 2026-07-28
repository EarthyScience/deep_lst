"""ERA5 LST error vs surface drivers — PER-REGION spatial/seasonal GAMs (mgcv bam), 2024.

One stable single-region model is fit per (region, response); results are reassembled for cross-region
comparison. (A single all-regions model with by=region smooths was rank-deficient / non-convergent;
per-region fits reuse the formula that fits cleanly and compare more directly.)

Pipeline (CLI subcommands -> separate SLURM stages):
    build         parallel per-region table build (multiprocessing) -> one parquet
    fit <0..17>   fit ONE (region, response): full/cov/space, save partition + summary + rds + curves
    aggregate     combine -> partition.csv (region x response a/b/c), terms.csv, curves.csv
    plot          overlay the per-region partial-effect curves per term/response

Per-region model (vegetated FVC>0, Gamma-log):
    rmse ~ s(fvc)+s(albedo)+s(dem)+s(amplitude) + s(month,bs='cc') + ti(fvc,month) + te(lon,lat)

Run via scripts/diagnostics/run_residual_analysis.sh. rpy2/R imported lazily so the build is R-free.
"""

import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from eodata.grids import T1H

from deep_lst.exploration.full_store import SEVIRI_STORE, load_store_columns
from deep_lst.utils.spatial import store_grid

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN-slice noise from cloudy pixels
N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

ERA5_PATTERN = "/Net/Groups/BGI/people/mwessel/deep_lst/assets/baselines/era5/eval_{}.nc"
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
RUN_NAME = "era5_allregions"
OUT_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/gam") / RUN_NAME
TABLE_PATH = OUT_DIR / "table.parquet"
YEAR = 2024
DAY_HOURS = (10, 15)  # inclusive UTC valid-hour window
NIGHT_HOURS = (22, 3)  # wraps midnight

SEVIRI_LOAD = {
    "data": ["LST"],
    "data_daily": ["FVC", "AL-BB-DH"],
    "data_static": ["DEM"],
    "data_static_categorical": ["LANDCOV"],
}
YEAR_DAY_RANGE = (T1H.date_to_index(f"{YEAR}-01-01") // 24, T1H.date_to_index(f"{YEAR + 1}-01-01") // 24)


# --- stage 1: build ----------------------------------------------------------
def region_table(region):
    """Tidy per-(pixel, month) table for one region: day/night ERA5 RMSE + drivers, for YEAR."""
    era5 = xr.open_dataset(ERA5_PATTERN.format(region))
    era5_time = pd.DatetimeIndex(era5["valid_time"].values)

    # Eval anchors ARE seviri pixels, so the full-store rows are the anchor ids themselves.
    columns, _, _, anchor_ids = load_store_columns(
        "seviri", SEVIRI_STORE, f"eval_{region}", SEVIRI_LOAD, anchor="seviri", day_range=YEAR_DAY_RANGE
    )
    lst_hourly = columns["data"].numpy()[:, :, 0]  # (n_pixel, n_hours)
    fvc_hourly = columns["data_daily"].numpy()[:, :, 0]  # daily values, repeated per hour
    albedo_hourly = columns["data_daily"].numpy()[:, :, 1]
    dem = columns["data_static"].numpy()[:, 0]
    landcov = columns["data_static_categorical"].numpy()[:, 0]
    centers = store_grid("seviri").centers[anchor_ids]  # (n_pixel, 2) lon/lat
    pix_lon, pix_lat = centers[:, 0], centers[:, 1]
    n_pixel = len(anchor_ids)
    store_time = pd.date_range(f"{YEAR}-01-01", periods=lst_hourly.shape[1], freq="h")

    grid_lat, grid_lon = era5["lat"].values, era5["lon"].values
    nearest_lat = np.abs(pix_lat[:, None] - grid_lat[None, :]).argmin(axis=1)
    nearest_lon = np.abs(pix_lon[:, None] - grid_lon[None, :]).argmin(axis=1)

    months = []
    for month in range(1, 13):
        era5_idx = np.where((era5_time.year == YEAR) & (era5_time.month == month))[0]
        month_idx = np.where(store_time.month == month)[0]
        if len(era5_idx) == 0 or len(era5_idx) != len(month_idx) or len(era5_idx) % 24 != 0:
            continue

        era5_pix = era5["skt"].isel(valid_time=era5_idx).values[:, nearest_lat, nearest_lon]
        lst = lst_hourly[:, month_idx].T  # (t, n_pixel)
        fvc = fvc_hourly[:, month_idx].T
        albedo = albedo_hourly[:, month_idx].T
        hour = era5_time[era5_idx].hour.values

        squared_error = (era5_pix - lst) ** 2
        day_mask = (hour >= DAY_HOURS[0]) & (hour <= DAY_HOURS[1])
        night_mask = (hour >= NIGHT_HOURS[0]) | (hour <= NIGHT_HOURS[1])
        n_day = len(era5_idx) // 24
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


def _region_table_named(region):
    return region, region_table(region)


def build():
    t0 = time.perf_counter()
    n_workers = min(len(REGIONS), max(N_THREADS, 1))
    print(f"building {len(REGIONS)} region tables with {n_workers} workers...", flush=True)
    tables = []
    with Pool(processes=n_workers) as pool:
        for region, tab in pool.imap_unordered(_region_table_named, REGIONS):
            tables.append(tab)
            print(f"  {region}: {len(tab)} rows ({time.perf_counter() - t0:.0f}s elapsed)", flush=True)
    table = pd.concat(tables, ignore_index=True)
    table["is_bare"] = (table["fvc"] == 0).astype(int)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TABLE_PATH)
    print(f"saved {TABLE_PATH}: {len(table)} rows ({time.perf_counter() - t0:.0f}s)", flush=True)


# --- stage 2: fit (mgcv via rpy2, lazy import) -------------------------------
_R = {}


def _r():
    if not _R:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        _R.update(
            ro=ro,
            pandas2ri=pandas2ri,
            localconverter=localconverter,
            mgcv=importr("mgcv"),
            grdevices=importr("grDevices"),
            family=ro.r('Gamma(link="log")'),
            knots=ro.r("list(month = c(0.5, 12.5))"),
            predict_terms=ro.r("function(m, nd) predict(m, newdata=nd, type='terms', se.fit=TRUE)"),
        )
    return _R


_SMOOTHS = "s(fvc) + s(albedo) + s(dem) + s(amplitude)"
MODELS = {
    "full": f"{{y}} ~ {_SMOOTHS} + s(month, bs='cc', k=6) + ti(fvc, month, bs=c('tp','cc'), k=c(6, 6)) + te(lon, lat)",
    "cov": f"{{y}} ~ {_SMOOTHS} + s(month, bs='cc', k=6) + ti(fvc, month, bs=c('tp','cc'), k=c(6, 6))",
    "space": "{y} ~ te(lon, lat)",
}
RESPONSES = ["rmse_day", "rmse_night"]
FIT_TASKS = [(region, response) for region in REGIONS for response in RESPONSES]  # 18 tasks
GAM_COLUMNS = ["fvc", "albedo", "dem", "amplitude", "month", "lon", "lat"]
CURVE_TERMS = ["fvc", "albedo", "dem", "amplitude", "month"]  # 1-D smooths exported for overlay
GRID_N = 100
MIN_ROWS = 500  # cells with fewer valid pixel-months are skipped (e.g. yangambi: permanent cloud -> 0)


def _fit(formula, r_df):
    # gam(REML), not bam(discrete): per-region data is small (~30k rows) and gam centres te() cleanly
    # (bam discrete triggered a te "reparameterization unstable" warning + an inflated intercept).
    r = _r()
    return r["mgcv"].gam(r["ro"].Formula(formula), data=r_df, family=r["family"], knots=r["knots"], method="REML")


def _dev_expl(fit):
    return float(_r()["ro"].r["summary"](fit).rx2("dev.expl")[0])


def _summary_text(fit):
    ro = _r()["ro"]
    return "\n".join(ro.r["capture.output"](ro.r["summary"](fit)))


def _smooth_table(fit, region, response):
    ro = _r()["ro"]
    s_table = ro.r["summary"](fit).rx2("s.table")
    df = pd.DataFrame(np.array(s_table), columns=list(ro.r["colnames"](s_table)))
    df.insert(0, "term", list(ro.r["rownames"](s_table)))
    df.insert(0, "region", region)
    df.insert(0, "response", response)
    return df


def _save_concurvity(fit, tag):
    r = _r()
    conc = r["mgcv"].concurvity(fit, full=True)
    df = pd.DataFrame(
        np.array(conc), index=list(r["ro"].r["rownames"](conc)), columns=list(r["ro"].r["colnames"](conc))
    )
    df.to_csv(OUT_DIR / f"concurvity_{tag}.csv")


def _save_partial_pdf(fit, tag):
    r = _r()
    r["grdevices"].pdf(str(OUT_DIR / f"partial_effects_{tag}.pdf"), width=12, height=8)
    r["ro"].r["plot"](fit, pages=0, scheme=2, shade=True, seWithMean=True)
    r["grdevices"].dev_off()


def _partial_curves(fit, data, region, response):
    """1-D smooth partial-effect curves on each region's OWN covariate range (no extrapolation beyond
    its data). type='terms' isolates each smooth (its contribution depends only on its own covariate),
    so the held values don't affect the curve. Returns long df: response, region, term, x, fit, se."""
    r = _r()
    holds = {c: float(np.nanmedian(data[c])) for c in GAM_COLUMNS}
    rows = []
    for term in CURVE_TERMS:
        if term == "month":
            grid = np.arange(1, 13, dtype=float)
        else:
            grid = np.linspace(float(data[term].min()), float(data[term].max()), GRID_N)
        newdata = pd.DataFrame({c: np.full(len(grid), holds[c]) for c in GAM_COLUMNS})
        newdata[term] = grid
        with r["localconverter"](r["ro"].default_converter + r["pandas2ri"].converter):
            nd = r["ro"].conversion.py2rpy(newdata)
        pred = r["predict_terms"](fit, nd)
        fit_mat = np.array(pred.rx2("fit"))
        se_mat = np.array(pred.rx2("se.fit"))
        cols = list(r["ro"].r["colnames"](pred.rx2("fit")))
        j = cols.index(f"s({term})")
        rows.append(
            pd.DataFrame(
                {
                    "response": response,
                    "region": region,
                    "term": term,
                    "x": grid,
                    "fit": fit_mat[:, j],
                    "se": se_mat[:, j],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def fit_cell(index):
    """Fit full/cov/space for one (region, response); save partition + summary + rds + curves."""
    region, response = FIT_TASKS[index]
    t0 = time.perf_counter()
    table = pd.read_parquet(TABLE_PATH)
    sub = (
        table[(table["region"] == region) & (table["fvc"] > 0) & (table[response] > 0)]
        .dropna(subset=["fvc", "albedo", "dem", "amplitude", response, "lon", "lat"])
        .copy()
    )

    if len(sub) < MIN_ROWS:  # not enough clear-sky data to fit (e.g. rainforest daytime) -> skip, don't fail
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"partition_{region}_{response}.json").write_text(
            json.dumps(
                {
                    "region": region,
                    "response": response,
                    "n": int(len(sub)),
                    "skipped": True,
                    "dev_full": None,
                    "dev_cov": None,
                    "dev_space": None,
                    "a_cov_unique": None,
                    "b_space_unique": None,
                    "c_shared": None,
                }
            )
        )
        print(f"[{index}] {region}_{response}: SKIPPED (only {len(sub)} valid rows < {MIN_ROWS})", flush=True)
        return

    r = _r()
    with r["localconverter"](r["ro"].default_converter + r["pandas2ri"].converter):
        r_df = r["ro"].conversion.py2rpy(sub[GAM_COLUMNS + [response]])
    fits = {m: _fit(MODELS[m].format(y=response), r_df) for m in MODELS}
    dev = {m: _dev_expl(fits[m]) for m in MODELS}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{region}_{response}"
    (OUT_DIR / f"gam_summary_{tag}.txt").write_text(_summary_text(fits["full"]))
    r["ro"].r["saveRDS"](fits["full"], file=str(OUT_DIR / f"gam_full_{tag}.rds"))
    _smooth_table(fits["full"], region, response).to_csv(OUT_DIR / f"terms_{tag}.csv", index=False)
    _save_concurvity(fits["full"], tag)
    _save_partial_pdf(fits["full"], tag)
    _partial_curves(fits["full"], sub, region, response).to_csv(OUT_DIR / f"curves_{tag}.csv", index=False)
    (OUT_DIR / f"partition_{tag}.json").write_text(
        json.dumps(
            {
                "region": region,
                "response": response,
                "n": int(len(sub)),
                "dev_full": dev["full"],
                "dev_cov": dev["cov"],
                "dev_space": dev["space"],
                "a_cov_unique": dev["full"] - dev["space"],
                "b_space_unique": dev["full"] - dev["cov"],
                "c_shared": dev["cov"] + dev["space"] - dev["full"],
            }
        )
    )
    print(
        f"[{index}] {tag}: n={len(sub)} dev full/cov/space={dev['full']:.3f}/{dev['cov']:.3f}/{dev['space']:.3f} ({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )


# --- stage 3: aggregate ------------------------------------------------------
def aggregate():
    # Iterate the known (region, response) tags so stale files from earlier runs aren't picked up.
    parts, terms, curves = [], [], []
    for region, response in FIT_TASKS:
        tag = f"{region}_{response}"
        if (OUT_DIR / f"partition_{tag}.json").exists():
            parts.append(json.loads((OUT_DIR / f"partition_{tag}.json").read_text()))
        if (OUT_DIR / f"terms_{tag}.csv").exists():
            terms.append(pd.read_csv(OUT_DIR / f"terms_{tag}.csv"))
        if (OUT_DIR / f"curves_{tag}.csv").exists():
            curves.append(pd.read_csv(OUT_DIR / f"curves_{tag}.csv"))
    pd.DataFrame(parts).to_csv(OUT_DIR / "partition.csv", index=False)
    if terms:
        pd.concat(terms, ignore_index=True).to_csv(OUT_DIR / "terms.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(OUT_DIR / "curves.csv", index=False)
    print(f"aggregated -> {OUT_DIR}/partition.csv, terms.csv, curves.csv", flush=True)


# --- stage 4: plot (overlay regions per term; pure python, no R) -------------
def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = pd.read_csv(OUT_DIR / "curves.csv")
    regions = sorted(curves["region"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {region: cmap(i % 10) for i, region in enumerate(regions)}

    for response in RESPONSES:
        terms = list(CURVE_TERMS)
        fig, axes = plt.subplots(1, len(terms), figsize=(4 * len(terms), 4.2), squeeze=False)
        for ax, term in zip(axes[0], terms):
            for region in regions:
                c = curves[(curves["response"] == response) & (curves["term"] == term) & (curves["region"] == region)]
                if c.empty:
                    continue
                ax.plot(c["x"], c["fit"], color=colors[region], linewidth=1.5, label=region)
                ax.fill_between(c["x"], c["fit"] - c["se"], c["fit"] + c["se"], color=colors[region], alpha=0.12)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
            ax.set_xlabel(term)
            ax.set_title(f"s({term})")
        axes[0][0].set_ylabel(f"partial effect on log {response}")
        axes[0][-1].legend(fontsize=7, ncol=1, loc="best")
        fig.suptitle(f"Per-region partial-effect curves — {response}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"curves_{response}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"saved {OUT_DIR}/curves_{response}.pdf", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("index", type=int, help="task index 0..17 (region x response)")
    sub.add_parser("aggregate")
    sub.add_parser("plot")
    args = parser.parse_args()

    if args.cmd == "build":
        build()
    elif args.cmd == "fit":
        fit_cell(args.index)
    elif args.cmd == "aggregate":
        aggregate()
    elif args.cmd == "plot":
        plot()
