"""Similarity of ERA5-Land fields and SEVIRI LST over Africa: SKT (hourly mode), swvl1 (daily mode).

Hourly mode (skt_lst): signed bias SKT - LST on three temporal scales (hourly, daily, monthly) and the Spearman rank
correlation in a rolling 10-day window, per 0.1 degree ERA5-Land cell of the African continent,
read from the full lowres_old stores. Clear-sky pairs only: the SEVIRI side counts as observed
when at least MIN_CLEAR of the 3 x 3 SEVIRI pixels around the cell centre carry a retrieval, the
ERA5 side when ERA5 total cloud cover is below TCC_MAX.

Daily mode (swvl1_lst, a different quantity, so no bias): daily daytime LST (mean of the clear-sky
hours at the local daytime peak DAY_HOURS) against the daily mean of the model field, as
    - a rolling DAILY_WINDOW_DAYS Spearman correlation in the analysis year, and
    - the "frois" co-variability score: per calendar month the interannual Spearman correlation
      of the monthly means over FROIS_YEARS, its Fisher-z p-value linearised to a weight
      w = 1 (p <= 0.01) .. 0 (p > 0.1), w = 0 below FROIS_MIN_YEARS years, and
      Ncovar = sum of the 12 monthly weights (0..12).

Layout (functional, top to bottom):
    SOURCES / PAIRS   which store field plays observation, model and cloud condition
    grids             target window, Africa polygon, alignment of a source grid to the target grid
    loading           one latitude band of hourly values per source, aligned to the target grid
    statistics        pairing and the per-band statistics of the four temporal products
    cubes             band processing, assembly and the NetCDF cubes (one per temporal scale)
    figures           the PDFs drawn from the cubes (nine in hourly mode, five in daily mode)

Transfer to another pair is a new PAIRS entry; a pair without a cloud condition sets cloud to
None. A daily-only model source (GloFAS swir) still needs a loader for daily store leaves.

    python deep_lst/exploration/era5.py compute --workers 8
    python deep_lst/exploration/era5.py compute --rows 300 308 --workers 1 --cube_dir /tmp/era5_smoke
    python deep_lst/exploration/era5.py plot
    python deep_lst/exploration/era5.py compute --pair swvl1_lst --cube_dir <dir>/swvl1_lst   # then plot --cube_dir ...
    python deep_lst/exploration/era5.py animate --start 2024-03-01 --days 3   # hourly SKT + LST gifs and pdf frames (slide)
"""

import argparse
import calendar
import io
import multiprocessing
import time
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from cartopy.io import shapereader
from eodata.grids import T1H
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.stats import norm, rankdata, spearmanr

from deep_lst.utils.full_store import ERA5_LAND_STORE, ERA5_STORE, SEVIRI_STORE, open_store_reader
from deep_lst.utils.spatial import store_grid

# %% ============================== CONFIG =============================================
YEAR = 2024
PAIR = "skt_lst"
CUBE_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/results/independent/era5")
FIG_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/independent/era5")

# window on the 0.1 degree ERA5-Land grid; cells outside the continent polygon are not processed
LAT_NORTH, LAT_SOUTH, LON_WEST, LON_EAST = 38.0, -36.0, -20.0, 55.0
TCC_MAX = 0.1  # ERA5 total cloud cover below which the model side counts as clear sky
MIN_CLEAR = 5  # SEVIRI pixels (of 9) with a retrieval for the cell to count as observed
WINDOW_DAYS = 10  # Spearman window, centred on each day
MIN_PAIRS = 24  # pairs per window below which the correlation is dropped
BAND_ROWS = 8  # target latitude rows per processing band
PAIRS_PER_BAND = 2000  # random hourly pairs kept per band for the density figure
DAY_HOURS = (12, 13, 14, 15)  # local solar hours of the daytime peak
NIGHT_HOURS = (2, 3, 4, 5)  # local solar hours of the night trough
DAILY_WINDOW_DAYS = 30  # daily mode: rolling Spearman window, centred on each day
DAILY_MIN_DAYS = 10  # daily mode: valid days per window below which the correlation is dropped
FROIS_YEARS = (2016, 2024)  # first and last year of the interannual correlation (SEVIRI store)
FROIS_MIN_DAYS = 5  # valid days for a monthly mean
FROIS_MIN_YEARS = 7  # the paper uses 10 (of 20 years); the SEVIRI store has at most 9
FROIS_P_STRONG, FROIS_P_WEAK = 0.01, 0.1  # p-values of weight 1 and weight 0

SOURCES = {
    "lst": {"modality": "seviri", "store": SEVIRI_STORE, "field": "LST"},
    "skt": {"modality": "era5_land", "store": ERA5_LAND_STORE, "field": "skt"},
    "swvl1": {"modality": "era5_land", "store": ERA5_LAND_STORE, "field": "swvl1"},
    "tcc": {"modality": "era5", "store": ERA5_STORE, "field": "tcc"},
}
# mode hourly: same quantity, bias on three scales + hourly rolling correlation;
# mode daily: different quantities, daily rolling correlation + the frois score.
PAIRS = {
    "skt_lst": {"obs": "lst", "model": "skt", "cloud": "tcc", "mode": "hourly"},
    "swvl1_lst": {"obs": "lst", "model": "swvl1", "cloud": None, "mode": "daily"},
}
LABELS = {
    "lst": ("SEVIRI LST", "K"),
    "skt": ("ERA5-Land SKT", "K"),
    "swvl1": ("ERA5-Land swvl1", "m$^3$ m$^{-3}$"),
}

MISSING_COLOR = "#ebebeb"
LABEL_COLOR = "#52514e"
BIAS_CMAP = "RdBu_r"

# animate: hourly SKT and SEVIRI LST of the continent as bare maps (time stamp + colorbar
# only, the layout of the split-window walkthrough's gifs) for a slide.
ANIMATION_SOURCES = ("skt", "lst")
# SKT on the whole ERA5-Land grid, LST on the analysis window of the continent (SEVIRI's reach).
ANIMATION_WINDOWS = {"skt": "global", "lst": "africa"}
GLOBAL_BAND_ROWS = 100  # latitude rows per contiguous global read (one worker task each)
ANIMATION_WORKERS = 8
ANIMATION_START = "2024-03-01"  # first day shown (UTC)
ANIMATION_DAYS = 3
ANIMATION_FPS = 8
ANIMATION_CMAP = "inferno"  # the LST colours of the split-window walkthrough (sw_algorithm.py)
ANIMATION_PDF_HOUR = 12  # UTC hour of the first day drawn as the static pdf frame
ANIMATION_INCHES = 7.0


# %% ============================== GRIDS ==============================================
def nearest_index(axis, values):
    """Index of the axis element closest to each value (axis equally spaced, any direction)."""
    step = axis[1] - axis[0]
    index = np.rint((np.asarray(values, dtype=float) - axis[0]) / step).astype(int)
    return np.clip(index, 0, len(axis) - 1)


def target_window(grid):
    """(row_start, row_stop), (col_start, col_stop) of the analysis window on the target grid."""
    rows = nearest_index(grid.y, [LAT_NORTH, LAT_SOUTH])
    cols = nearest_index(grid.x, [LON_WEST, LON_EAST])
    return (rows[0], rows[1] + 1), (cols[0], cols[1] + 1)


def africa_mask(lats, lons):
    """True for cell centres inside the Natural Earth (50 m) country polygons of Africa."""
    path = shapereader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
    records = shapereader.Reader(path).records()
    countries = [record.geometry for record in records if record.attributes["CONTINENT"] == "Africa"]
    continent = shapely.union_all(countries)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return shapely.contains_xy(continent, lon2d, lat2d)


def local_hours(lons):
    """Local solar hour of every UTC hour for each longitude: (n_lons, 24)."""
    shift = np.rint(np.asarray(lons) / 15.0).astype(int)
    return (np.arange(24)[None, :] + shift[:, None]) % 24


# %% ============================== LOADING ============================================
def open_source(source):
    """Store array, native grid and field name of one SOURCES entry."""
    reader = open_store_reader(source["modality"], source["store"], ["data"])
    return {"array": reader.data_arr, "grid": store_grid(source["modality"]), "field": source["field"]}


def read_hourly(opened, row_start, row_stop, day_start, day_stop):
    """(rows, days, 24) float32 of one field for a contiguous store row range."""
    block = opened["array"][row_start:row_stop, day_start:day_stop][opened["field"]]
    n_rows, n_days = row_stop - row_start, day_stop - day_start
    return np.asarray(block, dtype=np.float32).reshape(n_rows, n_days, 24)


def read_rows(opened, source_rows, col_start, col_stop, days):
    """(n_source_rows, cols, days, 24): one contiguous read per source latitude row."""
    n_lon = opened["grid"].get_shape()[1]
    blocks = []
    for source_row in source_rows:
        start = source_row * n_lon + col_start
        blocks.append(read_hourly(opened, start, start + (col_stop - col_start), *days))
    return np.stack(blocks)


def neighbourhood_mean(block, n_rows, n_cols, min_count):
    """Mean of the 3 x 3 source cells around every second source cell; NaN below min_count finite."""
    finite = np.isfinite(block)
    values = np.nan_to_num(block, copy=False)
    total = np.zeros((n_rows, n_cols) + block.shape[2:], dtype=np.float32)
    count = np.zeros((n_rows, n_cols) + block.shape[2:], dtype=np.int8)
    for row_offset in range(3):
        for col_offset in range(3):
            rows = slice(row_offset, row_offset + 2 * n_rows, 2)
            cols = slice(col_offset, col_offset + 2 * n_cols, 2)
            total += values[rows, cols]
            count += finite[rows, cols]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return np.where(count >= min_count, mean, np.nan).astype(np.float32)


def load_band(opened, lats, lons, days, min_clear):
    """Hourly values of one source on the target cells: (rows, cols, days, 24).

    Same resolution as the target: direct read. Coarser source: nearest source cell. Twice
    finer source (SEVIRI on ERA5-Land): mean of the 3 x 3 neighbourhood around the coincident
    centre, NaN where fewer than min_clear neighbours are finite.
    """
    grid = opened["grid"]
    ratio = grid.resolution / abs(lats[1] - lats[0])
    centre_rows = nearest_index(grid.y, lats)
    centre_cols = nearest_index(grid.x, lons)
    if np.isclose(ratio, 1.0):
        return read_rows(opened, centre_rows, centre_cols[0], centre_cols[-1] + 1, days)
    if ratio > 1.0:
        source_rows = np.unique(centre_rows)
        block = read_rows(opened, source_rows, centre_cols[0], centre_cols[-1] + 1, days)
        row_position = np.searchsorted(source_rows, centre_rows)
        return block[row_position][:, centre_cols - centre_cols[0]]
    if np.isclose(ratio, 0.5):
        assert np.all(np.diff(centre_rows) == 2) and np.all(np.diff(centre_cols) == 2)
        source_rows = np.arange(centre_rows[0] - 1, centre_rows[-1] + 2)
        block = read_rows(opened, source_rows, centre_cols[0] - 1, centre_cols[-1] + 2, days)
        return neighbourhood_mean(block, len(lats), len(lons), min_clear)
    raise ValueError(f"unsupported resolution ratio {ratio:.3f} between source and target grid")


# %% ============================== STATISTICS =========================================
def pair_mask(obs, model, cloud, tcc_max):
    """True where both sides are finite and (if given) the cloud condition holds."""
    valid = np.isfinite(obs) & np.isfinite(model)
    if cloud is not None:
        valid &= cloud < tcc_max
    return valid


def binned_moments(bias, valid, bin_index, n_bins):
    """Mean, std, RMSE and count of the valid bias values per cell and bin: (cells, n_bins)."""
    n_cells = bias.shape[0]
    cell_index = np.arange(n_cells, dtype=np.int64)[:, None, None]
    flat_index = (cell_index * n_bins + bin_index)[valid]
    values = bias[valid].astype(np.float64)
    size = n_cells * n_bins
    total = np.bincount(flat_index, weights=values, minlength=size)
    squares = np.bincount(flat_index, weights=values * values, minlength=size)
    count = np.bincount(flat_index, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
        mean_square = squares / count
        std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
        rmse = np.sqrt(mean_square)
    shape = (n_cells, n_bins)
    return {
        "bias_mean": mean.reshape(shape).astype(np.float32),
        "bias_std": std.reshape(shape).astype(np.float32),
        "rmse": rmse.reshape(shape).astype(np.float32),
        "n_pairs": count.reshape(shape).astype(np.int32),
    }


def hourly_moments(bias, valid, months, cell_local_hours):
    """Bias moments per cell, month and local solar hour: dict of (cells, 12, 24)."""
    bin_index = (months[None, :, None] - 1) * 24 + cell_local_hours[:, None, :]
    moments = binned_moments(bias, valid, bin_index.astype(np.int64), 12 * 24)
    return {key: value.reshape(bias.shape[0], 12, 24) for key, value in moments.items()}


def monthly_moments(bias, valid, months):
    """Bias moments per cell and month: dict of (cells, 12)."""
    bin_index = np.broadcast_to((months - 1)[None, :, None], bias.shape).astype(np.int64)
    return binned_moments(bias, valid, bin_index, 12)


def daily_extrema(obs, model, bias, valid):
    """Per cell and day: bias of the daily mean, maximum and minimum over the paired hours."""
    obs_valid = np.where(valid, obs, np.nan)
    model_valid = np.where(valid, model, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days are NaN by design
        bias_mean = np.nanmean(np.where(valid, bias, np.nan), axis=2)
        bias_max = np.nanmax(model_valid, axis=2) - np.nanmax(obs_valid, axis=2)
        bias_min = np.nanmin(model_valid, axis=2) - np.nanmin(obs_valid, axis=2)
    return {
        "bias_mean": bias_mean.astype(np.float32),
        "bias_max": bias_max.astype(np.float32),
        "bias_min": bias_min.astype(np.float32),
        "n_hours": valid.sum(axis=2).astype(np.int16),
    }


def monthly_mean_of_daily(daily, months):
    """(cells, 12) nanmean over the days of each month of a (cells, days) array."""
    out = np.full((daily.shape[0], 12), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for month in range(1, 13):
            out[:, month - 1] = np.nanmean(daily[:, months == month], axis=1)
    return out


def pearson_rows(a, b):
    """Pearson correlation along axis 1 over the entries finite in both: (rho, n)."""
    both = np.isfinite(a) & np.isfinite(b)
    n = both.sum(axis=1)
    a = np.where(both, a, 0.0)
    b = np.where(both, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_centred = np.where(both, a - (a.sum(axis=1) / n)[:, None], 0.0)
        b_centred = np.where(both, b - (b.sum(axis=1) / n)[:, None], 0.0)
        covariance = (a_centred * b_centred).sum(axis=1)
        rho = covariance / np.sqrt((a_centred**2).sum(axis=1) * (b_centred**2).sum(axis=1))
    return rho, n


def rolling_spearman(obs, model, valid, window_days, min_pairs):
    """Spearman correlation of the paired hours in a window centred on each day: (cells, days)."""
    n_cells, n_days, n_hours = obs.shape
    obs_flat = np.where(valid, obs, np.nan).reshape(n_cells, n_days * n_hours)
    model_flat = np.where(valid, model, np.nan).reshape(n_cells, n_days * n_hours)
    rho = np.full((n_cells, n_days), np.nan, dtype=np.float32)
    count = np.zeros((n_cells, n_days), dtype=np.int32)
    half = window_days // 2
    for day in range(n_days):
        start = max(0, day - half) * n_hours
        stop = min(n_days, day + window_days - half) * n_hours
        obs_rank = rankdata(obs_flat[:, start:stop], axis=1, nan_policy="omit")
        model_rank = rankdata(model_flat[:, start:stop], axis=1, nan_policy="omit")
        rho[:, day], count[:, day] = pearson_rows(obs_rank, model_rank)
    rho[count < min_pairs] = np.nan
    return {"rho": rho, "n_pairs": count}


def sample_pairs(obs, model, valid, cell_local_hours, cell_lats, cell_lons, n_samples, seed):
    """Random subset of the valid hourly pairs with their local hour and position."""
    cell, day, hour = np.nonzero(valid)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cell), size=min(n_samples, len(cell)), replace=False)
    cell, day, hour = cell[pick], day[pick], hour[pick]
    return {
        "obs": obs[cell, day, hour],
        "model": model[cell, day, hour],
        "local_hour": cell_local_hours[cell, hour].astype(np.int8),
        "pair_lat": cell_lats[cell].astype(np.float32),
        "pair_lon": cell_lons[cell].astype(np.float32),
    }


# %% ============================== CUBES ==============================================
WORKER = {}  # opened sources and static inputs of the band workers (set by init_worker)


def init_worker(pair, days, lons, continent, months, tcc_max, min_clear):
    names = [pair["obs"], pair["model"]] + ([pair["cloud"]] if pair["cloud"] else [])
    WORKER["opened"] = {name: open_source(SOURCES[name]) for name in names}
    WORKER["pair"] = pair
    WORKER["days"] = days
    WORKER["lons"] = lons
    WORKER["continent"] = continent
    WORKER["months"] = months
    WORKER["tcc_max"] = tcc_max
    WORKER["min_clear"] = min_clear


def process_band(band):
    """All per-cell products of one band of target rows; arrays are over the band's cells."""
    row_start, row_stop = band
    started = time.time()
    opened, pair, lons = WORKER["opened"], WORKER["pair"], WORKER["lons"]
    lats = WORKER["lats"][row_start:row_stop]
    n_rows, n_cols = len(lats), len(lons)
    n_cells = n_rows * n_cols

    def load(name):
        field = load_band(opened[name], lats, lons, WORKER["days"], WORKER["min_clear"])
        return field.reshape(n_cells, *field.shape[2:])

    obs = load(pair["obs"])
    model = load(pair["model"])
    cloud = load(pair["cloud"]) if pair["cloud"] else None
    valid = pair_mask(obs, model, cloud, WORKER["tcc_max"])
    valid &= WORKER["continent"][row_start:row_stop].reshape(n_cells)[:, None, None]
    active = np.nonzero(valid.any(axis=(1, 2)))[0]

    cell_lats = np.repeat(lats, n_cols)
    cell_lons = np.tile(lons, n_rows)
    cell_local_hours = local_hours(cell_lons)
    months = WORKER["months"]
    results = {"band": band, "n_cells": n_cells, "active": active}
    if len(active):
        obs, model, valid = obs[active], model[active], valid[active]
        bias = model - obs
        results["hourly"] = hourly_moments(bias, valid, months, cell_local_hours[active])
        results["daily"] = daily_extrema(obs, model, bias, valid)
        results["monthly"] = monthly_moments(bias, valid, months)
        results["monthly"]["bias_max_mean"] = monthly_mean_of_daily(results["daily"]["bias_max"], months)
        results["monthly"]["bias_min_mean"] = monthly_mean_of_daily(results["daily"]["bias_min"], months)
        results["continent_hourly"] = {
            "sum": np.where(valid, bias, 0.0).sum(axis=0).ravel(),
            "count": valid.sum(axis=0).ravel(),
        }
        results["spearman"] = rolling_spearman(obs, model, valid, WINDOW_DAYS, MIN_PAIRS)
        results["pairs"] = sample_pairs(
            obs, model, valid, cell_local_hours[active], cell_lats[active], cell_lons[active], PAIRS_PER_BAND, row_start
        )
    print(f"  band rows {row_start}-{row_stop}: {len(active)} active cells, {time.time() - started:.0f} s", flush=True)
    return results


def compute(args):
    pair = PAIRS[args.pair]
    if pair["mode"] == "daily":
        compute_daily(args, pair)
        return
    grid = store_grid(SOURCES[pair["model"]]["modality"])
    (row_start, row_stop), (col_start, col_stop) = target_window(grid)
    if args.rows is not None:
        row_start, row_stop = row_start + args.rows[0], row_start + args.rows[1]
    lats = grid.y[row_start:row_stop]
    lons = grid.x[col_start:col_stop]
    n_rows, n_cols = len(lats), len(lons)
    n_cells = n_rows * n_cols
    continent = africa_mask(lats, lons)
    day_start = T1H.date_to_index(f"{args.year}-01-01") // 24
    day_stop = T1H.date_to_index(f"{args.year + 1}-01-01") // 24
    dates = pd.date_range(f"{args.year}-01-01", periods=day_stop - day_start, freq="D")
    months = dates.month.to_numpy()
    n_days = len(dates)
    print(
        f"pair {args.pair}, {args.year}: {n_rows} x {n_cols} cells, {int(continent.sum())} in the continent polygon, "
        f"{n_days} days, bands of {BAND_ROWS} rows, {args.workers} workers",
        flush=True,
    )

    variables = {
        "hourly": (
            {"bias_mean": np.float32, "bias_std": np.float32, "rmse": np.float32, "n_pairs": np.int32},
            (12, 24),
        ),
        "daily": (
            {"bias_mean": np.float32, "bias_max": np.float32, "bias_min": np.float32, "n_hours": np.int16},
            (n_days,),
        ),
        "monthly": (
            {
                "bias_mean": np.float32,
                "bias_std": np.float32,
                "rmse": np.float32,
                "n_pairs": np.int32,
                "bias_max_mean": np.float32,
                "bias_min_mean": np.float32,
            },
            (12,),
        ),
        "spearman": ({"rho": np.float32, "n_pairs": np.int32}, (n_days,)),
    }
    cubes = {}
    for product, (names, trailing) in variables.items():
        cubes[product] = {}
        for name, dtype in names.items():
            fill = np.nan if dtype == np.float32 else 0
            cubes[product][name] = np.full((n_cells,) + trailing, fill, dtype=dtype)
    continent_sum = np.zeros(n_days * 24)
    continent_count = np.zeros(n_days * 24, dtype=np.int64)
    pairs = []

    bands = [(start, min(start + BAND_ROWS, n_rows)) for start in range(0, n_rows, BAND_ROWS)]
    init_args = (pair, (day_start, day_stop), lons, continent, months, args.tcc_max, args.min_clear)
    WORKER["lats"] = lats  # inherited by forked workers

    def collect(result):
        if "spearman" not in result:
            return
        offset = result["band"][0] * n_cols
        cells = offset + result["active"]
        for product in variables:
            for name, values in result[product].items():
                cubes[product][name][cells] = values
        continent_sum[:] += result["continent_hourly"]["sum"]
        continent_count[:] += result["continent_hourly"]["count"]
        pairs.append(result["pairs"])

    started = time.time()
    if args.workers == 1:
        init_worker(*init_args)
        for band in bands:
            collect(process_band(band))
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(args.workers, initializer=init_worker, initargs=init_args) as pool:
            for result in pool.imap_unordered(process_band, bands):
                collect(result)
    print(f"all bands done in {(time.time() - started) / 60:.1f} min", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        continent_hourly = continent_sum / continent_count
    write_cubes(args, cubes, (lats, lons), dates, continent_hourly, continent_count, pairs, continent, (n_rows, n_cols))


def continent_series(field, axis_name):
    """Mean over cells and number of contributing cells of a (cells, time) product variable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(field, axis=0)
    count = np.isfinite(field).sum(axis=0)
    return xr.DataArray(mean, dims=(axis_name,)), xr.DataArray(count, dims=(axis_name,))


def to_dataset(product, coords, shape, dims):
    """Dataset of one product: (cells, ...) arrays -> (..., lat, lon), with the coords of its dims."""
    data = {}
    for name, values in product.items():
        spatial = values.reshape(shape + values.shape[1:])
        data[name] = xr.DataArray(np.moveaxis(spatial, (0, 1), (-2, -1)), dims=dims + ("lat", "lon"))
    return xr.Dataset(data, coords={name: coords[name] for name in dims + ("lat", "lon")})


def write_cubes(args, cubes, axes, dates, continent_hourly, continent_count, pairs, continent, shape):
    lats, lons = axes
    args.cube_dir.mkdir(parents=True, exist_ok=True)
    hours = pd.date_range(dates[0], periods=len(dates) * 24, freq="h")
    coords = {
        "lat": lats.astype(np.float32),
        "lon": lons.astype(np.float32),
        "month": np.arange(1, 13),
        "hour": np.arange(24),
        "day": dates,
        "time": hours,
    }
    pair = PAIRS[args.pair]
    clear_sky = f">= {args.min_clear} of 9 SEVIRI pixels observed"
    if pair["cloud"]:
        clear_sky = f"{pair['cloud']} < {args.tcc_max} and {clear_sky}"
    attrs = {
        "pair": args.pair,
        "year": args.year,
        "sign_convention": f"bias = {pair['model']} - {pair['obs']}",
        "clear_sky": clear_sky,
        "hour": "local solar hour = UTC hour + round(lon / 15)",
        "spearman_window": f"{WINDOW_DAYS} days centred on the day, >= {MIN_PAIRS} pairs",
        "series": "hourly clear-sky",
        "window_days": WINDOW_DAYS,
        "min_pairs": MIN_PAIRS,
        "sources": ", ".join(f"{name}: {SOURCES[name]['store']} {SOURCES[name]['field']}" for name in SOURCES),
        "mask": "Natural Earth 50m countries, continent Africa; NaN outside",
    }
    encoding = {"zlib": True, "complevel": 4}
    datasets = bias_datasets(cubes, coords, shape, hours, continent_hourly, continent_count)
    spearman = to_dataset(cubes["spearman"], coords, shape, ("day",))
    spearman["continent_rho"], spearman["continent_n_cells"] = continent_series(cubes["spearman"]["rho"], "day")
    spearman = spearman.rename({"day": "window"})
    sample = xr.Dataset(
        {key: ("pair", np.concatenate([p[key] for p in pairs])) for key in pairs[0]},
        coords={"lat": coords["lat"], "lon": coords["lon"]},
    )
    sample["in_continent"] = xr.DataArray(continent, dims=("lat", "lon"))
    datasets += [("spearman_10d", spearman), ("pairs_sample", sample)]

    for name, dataset in datasets:
        dataset.attrs = attrs
        path = args.cube_dir / f"{name}.nc"
        dataset.to_netcdf(path, encoding={var: encoding for var in dataset.data_vars})
        print(f"wrote {path}", flush=True)


def bias_datasets(cubes, coords, shape, hours, continent_hourly, continent_count):
    """(name, Dataset) of the hourly, daily and monthly bias cubes."""
    hourly = to_dataset(cubes["hourly"], coords, shape, ("month", "hour"))
    hourly["continent_bias"] = xr.DataArray(continent_hourly.astype(np.float32), dims=("time",), coords={"time": hours})
    hourly["continent_n_cells"] = xr.DataArray(continent_count.astype(np.int32), dims=("time",), coords={"time": hours})
    daily = to_dataset(cubes["daily"], coords, shape, ("day",))
    for name in ("bias_mean", "bias_max", "bias_min"):
        daily[f"continent_{name}"], daily["continent_n_cells"] = continent_series(cubes["daily"][name], "day")
    monthly = to_dataset(cubes["monthly"], coords, shape, ("month",))
    monthly["continent_bias"], monthly["continent_n_cells"] = continent_series(cubes["monthly"]["bias_mean"], "month")
    return [("hourly", hourly), ("daily", daily), ("monthly", monthly)]


# %% ============================== DAILY MODE =========================================
def daytime_daily(obs, cell_local_hours, day_hours):
    """Mean of the finite values at the local daytime peak hours per cell and day: (cells, days)."""
    is_peak = np.isin(cell_local_hours, day_hours)  # (cells, 24)
    peak = np.where(is_peak[:, None, :], obs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # days without a clear peak hour are NaN
        return np.nanmean(peak, axis=2).astype(np.float32)


def daily_mean(model):
    """Mean over the 24 hours per cell and day: (cells, days)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(model, axis=2).astype(np.float32)


def monthly_means(daily, months, min_days):
    """(cells, 12) mean over the days of each month; NaN with fewer than min_days finite days."""
    out = np.full((daily.shape[0], 12), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for month in range(1, 13):
            in_month = daily[:, months == month]
            mean = np.nanmean(in_month, axis=1)
            enough = np.isfinite(in_month).sum(axis=1) >= min_days
            out[:, month - 1] = np.where(enough, mean, np.nan)
    return out


def interannual_spearman(obs_monthly, model_monthly):
    """Spearman correlation across years per cell and calendar month.

    Inputs (cells, years, 12); returns rho and the number of years (cells, 12).
    """
    both = np.isfinite(obs_monthly) & np.isfinite(model_monthly)
    obs_paired = np.where(both, obs_monthly, np.nan)
    model_paired = np.where(both, model_monthly, np.nan)
    n_cells = obs_monthly.shape[0]
    rho = np.full((n_cells, 12), np.nan, dtype=np.float32)
    n_years = both.sum(axis=1).astype(np.int16)
    for month in range(12):
        obs_rank = rankdata(obs_paired[:, :, month], axis=1, nan_policy="omit")
        model_rank = rankdata(model_paired[:, :, month], axis=1, nan_policy="omit")
        rho[:, month], _ = pearson_rows(obs_rank, model_rank)
    return rho, n_years


def fisher_pvalue(rho, n):
    """Two-sided p-value of a correlation against zero from the Fisher z-transformation."""
    rho_clipped = np.clip(rho, -0.999999, 0.999999)
    z = np.arctanh(rho_clipped) * np.sqrt(np.maximum(n - 3, 0))
    return (2.0 * norm.sf(np.abs(z))).astype(np.float32)


def frois_weight(p_value, n_years, min_years):
    """Eq. 1: 1 for p <= 0.01, 0 for p > 0.1, linear between; 0 below min_years or without rho."""
    weight = (FROIS_P_WEAK - p_value) / (FROIS_P_WEAK - FROIS_P_STRONG)
    weight = np.clip(weight, 0.0, 1.0)
    usable = (n_years >= min_years) & np.isfinite(p_value)
    return np.where(usable, weight, 0.0).astype(np.float32)


def sample_daily_pairs(obs, model, valid, cell_lats, cell_lons, n_samples, seed):
    """Random subset of the valid daily pairs with their position."""
    cell, day = np.nonzero(valid)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cell), size=min(n_samples, len(cell)), replace=False)
    cell, day = cell[pick], day[pick]
    return {
        "obs": obs[cell, day],
        "model": model[cell, day],
        "pair_lat": cell_lats[cell].astype(np.float32),
        "pair_lon": cell_lons[cell].astype(np.float32),
    }


def process_band_daily(band):
    """Daily series of all FROIS_YEARS, then the rolling correlation and the frois score."""
    row_start, row_stop = band
    started = time.time()
    opened, pair, lons = WORKER["opened"], WORKER["pair"], WORKER["lons"]
    lats = WORKER["lats"][row_start:row_stop]
    n_rows, n_cols = len(lats), len(lons)
    n_cells = n_rows * n_cols
    cell_lats = np.repeat(lats, n_cols)
    cell_lons = np.tile(lons, n_rows)
    cell_local_hours = local_hours(cell_lons)
    in_continent = WORKER["continent"][row_start:row_stop].reshape(n_cells)

    obs_monthly, model_monthly = [], []
    for year in WORKER["years"]:
        days = (year["day_start"], year["day_stop"])
        obs = load_band(opened[pair["obs"]], lats, lons, days, WORKER["min_clear"]).reshape(n_cells, -1, 24)
        model = load_band(opened[pair["model"]], lats, lons, days, WORKER["min_clear"]).reshape(n_cells, -1, 24)
        obs_daily = daytime_daily(obs, cell_local_hours, DAY_HOURS)
        model_daily = daily_mean(model)
        valid = np.isfinite(obs_daily) & np.isfinite(model_daily) & in_continent[:, None]
        obs_daily = np.where(valid, obs_daily, np.nan)
        model_daily = np.where(valid, model_daily, np.nan)
        obs_monthly.append(monthly_means(obs_daily, year["months"], FROIS_MIN_DAYS))
        model_monthly.append(monthly_means(model_daily, year["months"], FROIS_MIN_DAYS))
        if year["year"] == WORKER["analysis_year"]:
            analysis = (obs_daily, model_daily, valid)

    obs_daily, model_daily, valid = analysis
    results = {"band": band}
    rolling = rolling_spearman(
        obs_daily[:, :, None], model_daily[:, :, None], valid[:, :, None], DAILY_WINDOW_DAYS, DAILY_MIN_DAYS
    )
    results["spearman"] = rolling
    rho, n_years = interannual_spearman(np.stack(obs_monthly, axis=1), np.stack(model_monthly, axis=1))
    p_value = fisher_pvalue(rho, n_years)
    weight = frois_weight(p_value, n_years, WORKER["min_years"])
    results["frois"] = {"rho": rho, "p_value": p_value, "n_years": n_years, "weight": weight}
    results["frois_total"] = {"n_covar": weight.sum(axis=1).astype(np.float32)}
    results["frois_total"]["n_covar"][~in_continent] = np.nan
    results["pairs"] = sample_daily_pairs(
        obs_daily, model_daily, valid, cell_lats, cell_lons, PAIRS_PER_BAND, row_start
    )
    n_active = int(valid.any(axis=1).sum())
    print(f"  band rows {row_start}-{row_stop}: {n_active} active cells, {time.time() - started:.0f} s", flush=True)
    return results


def year_axes(first_year, last_year):
    """Store day range and calendar month of every day for each year."""
    years = []
    for year in range(first_year, last_year + 1):
        day_start = T1H.date_to_index(f"{year}-01-01") // 24
        day_stop = T1H.date_to_index(f"{year + 1}-01-01") // 24
        dates = pd.date_range(f"{year}-01-01", periods=day_stop - day_start, freq="D")
        years.append({"year": year, "day_start": day_start, "day_stop": day_stop, "months": dates.month.to_numpy()})
    return years


def compute_daily(args, pair):
    grid = store_grid(SOURCES[pair["model"]]["modality"])
    (row_start, row_stop), (col_start, col_stop) = target_window(grid)
    if args.rows is not None:
        row_start, row_stop = row_start + args.rows[0], row_start + args.rows[1]
    lats = grid.y[row_start:row_stop]
    lons = grid.x[col_start:col_stop]
    n_rows, n_cols = len(lats), len(lons)
    n_cells = n_rows * n_cols
    continent = africa_mask(lats, lons)
    years = year_axes(*FROIS_YEARS)
    analysis = [year for year in years if year["year"] == args.year]
    assert analysis, f"analysis year {args.year} outside FROIS_YEARS {FROIS_YEARS}"
    dates = pd.date_range(f"{args.year}-01-01", periods=len(analysis[0]["months"]), freq="D")
    n_days = len(dates)
    print(
        f"pair {args.pair} (daily mode), {args.year} + frois {FROIS_YEARS[0]}-{FROIS_YEARS[1]}: "
        f"{n_rows} x {n_cols} cells, {int(continent.sum())} in the continent polygon, "
        f"bands of {BAND_ROWS} rows, {args.workers} workers",
        flush=True,
    )

    variables = {
        "spearman": ({"rho": np.float32, "n_pairs": np.int32}, (n_days,)),
        "frois": ({"rho": np.float32, "p_value": np.float32, "n_years": np.int16, "weight": np.float32}, (12,)),
        "frois_total": ({"n_covar": np.float32}, ()),
    }
    cubes = {}
    for product, (names, trailing) in variables.items():
        cubes[product] = {}
        for name, dtype in names.items():
            fill = np.nan if dtype == np.float32 else 0
            cubes[product][name] = np.full((n_cells,) + trailing, fill, dtype=dtype)
    pairs = []

    bands = [(start, min(start + BAND_ROWS, n_rows)) for start in range(0, n_rows, BAND_ROWS)]
    init_args = (pair, None, lons, continent, None, args.tcc_max, args.min_clear)
    WORKER["lats"] = lats  # inherited by forked workers
    WORKER["years"] = years
    WORKER["analysis_year"] = args.year
    WORKER["min_years"] = args.min_years

    def collect(result):
        cells = result["band"][0] * n_cols + np.arange(len(result["frois_total"]["n_covar"]))
        for product in variables:
            for name, values in result[product].items():
                cubes[product][name][cells] = values
        pairs.append(result["pairs"])

    started = time.time()
    if args.workers == 1:
        init_worker(*init_args)
        for band in bands:
            collect(process_band_daily(band))
    else:
        context = multiprocessing.get_context("fork")
        with context.Pool(args.workers, initializer=init_worker, initargs=init_args) as pool:
            for result in pool.imap_unordered(process_band_daily, bands):
                collect(result)
    print(f"all bands done in {(time.time() - started) / 60:.1f} min", flush=True)
    write_daily_cubes(args, pair, cubes, (lats, lons), dates, pairs, continent, (n_rows, n_cols))


def write_daily_cubes(args, pair, cubes, axes, dates, pairs, continent, shape):
    lats, lons = axes
    args.cube_dir.mkdir(parents=True, exist_ok=True)
    coords = {"lat": lats.astype(np.float32), "lon": lons.astype(np.float32), "month": np.arange(1, 13), "day": dates}
    attrs = {
        "pair": args.pair,
        "year": args.year,
        "daily_obs": f"mean of the clear-sky {pair['obs']} hours at local solar hours {DAY_HOURS}",
        "daily_model": f"mean of {pair['model']} over the 24 hours",
        "clear_sky": f">= {args.min_clear} of 9 SEVIRI pixels observed",
        "hour": "local solar hour = UTC hour + round(lon / 15)",
        "series": "daily",
        "window_days": DAILY_WINDOW_DAYS,
        "min_pairs": DAILY_MIN_DAYS,
        "spearman_window": f"{DAILY_WINDOW_DAYS} days centred on the day, >= {DAILY_MIN_DAYS} valid days",
        "frois": (
            f"interannual Spearman per calendar month of monthly means (>= {FROIS_MIN_DAYS} valid days) over "
            f"{FROIS_YEARS[0]}-{FROIS_YEARS[1]}; Fisher-z two-sided p; weight 1 at p <= {FROIS_P_STRONG}, "
            f"0 at p > {FROIS_P_WEAK}, linear between, 0 below {args.min_years} years; n_covar = sum of 12 weights"
        ),
        "sources": ", ".join(f"{name}: {SOURCES[name]['store']} {SOURCES[name]['field']}" for name in SOURCES),
        "mask": "Natural Earth 50m countries, continent Africa; NaN outside",
    }
    encoding = {"zlib": True, "complevel": 4}

    spearman = to_dataset(cubes["spearman"], coords, shape, ("day",))
    spearman["continent_rho"], spearman["continent_n_cells"] = continent_series(cubes["spearman"]["rho"], "day")
    spearman = spearman.rename({"day": "window"})
    frois = to_dataset(cubes["frois"], coords, shape, ("month",))
    frois["n_covar"] = to_dataset(cubes["frois_total"], coords, shape, ())["n_covar"]
    sample = xr.Dataset(
        {key: ("pair", np.concatenate([p[key] for p in pairs])) for key in pairs[0]},
        coords={"lat": coords["lat"], "lon": coords["lon"]},
    )
    sample["in_continent"] = xr.DataArray(continent, dims=("lat", "lon"))

    for name, dataset in [("spearman_daily", spearman), ("frois", frois), ("pairs_sample", sample)]:
        dataset.attrs = attrs
        path = args.cube_dir / f"{name}.nc"
        dataset.to_netcdf(path, encoding={var: encoding for var in dataset.data_vars})
        print(f"wrote {path}", flush=True)


# %% ============================== FIGURES ============================================
def open_cubes(cube_dir):
    """The cubes found in cube_dir; the rolling correlation (hourly or daily) under 'spearman'."""
    files = {
        "hourly": "hourly",
        "daily": "daily",
        "monthly": "monthly",
        "spearman": "spearman_10d",
        "spearman_daily": "spearman_daily",
        "frois": "frois",
        "pairs_sample": "pairs_sample",
    }
    cubes = {}
    for key, stem in files.items():
        path = cube_dir / f"{stem}.nc"
        if path.exists():
            cubes["spearman" if key == "spearman_daily" else key] = xr.open_dataset(path)
    return cubes


def series_label(cubes):
    """Series and window of the rolling correlation; the first skt_lst cubes predate the attrs."""
    attrs = cubes["spearman"].attrs
    return (
        attrs.get("series", "hourly clear-sky"),
        attrs.get("window_days", WINDOW_DAYS),
        attrs.get("min_pairs", MIN_PAIRS),
    )


def pair_names(cubes):
    """(obs name, model name) of the pair the cubes were computed for."""
    pair = PAIRS[cubes["spearman"].attrs["pair"]]
    return LABELS[pair["obs"]][0], LABELS[pair["model"]][0]


def symmetric_limit(fields, quantile=0.98):
    """Colour limit for diverging maps: upper quantile of |values| over all fields, rounded up."""
    values = np.concatenate([np.abs(np.asarray(field).ravel()) for field in fields])
    values = values[np.isfinite(values)]
    return float(np.ceil(np.quantile(values, quantile))) if len(values) else 1.0


def draw_map(ax, field, cmap, vmin, vmax):
    """One (lat, lon) field as an image with coastlines; grey where NaN."""
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(MISSING_COLOR)
    lon, lat = field["lon"].values, field["lat"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    image = ax.imshow(
        field.values,
        extent=extent,
        origin="upper",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )
    ax.coastlines(lw=0.4)
    ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="grey")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    return image


def map_figure(panels, title, path, ncols, figsize, shrink=0.6):
    """Grid of maps. panels: list of (field, title, cmap, vmin, vmax, colorbar label); one
    colorbar per distinct label, placed next to the last panel carrying it."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    images = {}
    for ax, (field, panel_title, cmap, vmin, vmax, label) in zip(axes, panels):
        images[label] = (draw_map(ax, field, cmap, vmin, vmax), ax)
        ax.set_title(panel_title, fontsize=11, color=LABEL_COLOR)
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    shared = len(images) == 1
    for label, (image, ax) in images.items():
        colorbar = fig.colorbar(image, ax=axes[: len(panels)] if shared else ax, shrink=shrink, pad=0.02)
        colorbar.set_label(label, color=LABEL_COLOR)
        colorbar.outline.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def annual_from_monthly(monthly):
    """Count-weighted annual mean, std and RMSE of the bias from the monthly moments."""
    count = monthly["n_pairs"].astype(np.float64)
    total = (monthly["bias_mean"] * count).sum("month")
    squares = (monthly["rmse"] ** 2 * count).sum("month")
    n = count.sum("month").where(count.sum("month") > 0)
    mean = total / n
    mean_square = squares / n
    std = np.sqrt((mean_square - mean**2).clip(min=0.0))
    return {"bias_mean": mean, "bias_std": std, "rmse": np.sqrt(mean_square), "n_pairs": n}


def weighted_hour_mean(hourly, hours):
    """Count-weighted mean bias over the given local hours and all months: (lat, lon)."""
    subset = hourly.sel(hour=list(hours))
    count = subset["n_pairs"].astype(np.float64)
    total = (subset["bias_mean"] * count).sum(("month", "hour"))
    n = count.sum(("month", "hour"))
    return (total / n.where(n > 0)).where(n >= MIN_PAIRS)


def figure_bias_annual(cubes, fig_dir):
    annual = annual_from_monthly(cubes["monthly"])
    limit = symmetric_limit([annual["bias_mean"]])
    spread_max = float(np.nanquantile(annual["rmse"].values, 0.98))
    panels = [
        (annual["bias_mean"], "mean bias", BIAS_CMAP, -limit, limit, "SKT - LST [K]"),
        (annual["bias_std"], "std of the hourly bias", "magma", 0, spread_max, "[K]"),
        (annual["rmse"], "RMSE", "magma", 0, spread_max, "[K]"),
        (
            annual["n_pairs"],
            "clear-sky pairs",
            "Blues",
            0,
            float(np.nanquantile(annual["n_pairs"].values, 0.98)),
            "hours",
        ),
    ]
    map_figure(
        panels,
        f"ERA5-Land SKT vs SEVIRI LST, clear-sky hourly pairs {YEAR}",
        fig_dir / "bias_annual_maps.pdf",
        2,
        (12, 11),
    )


def figure_bias_monthly(cubes, fig_dir):
    monthly = cubes["monthly"]["bias_mean"].where(cubes["monthly"]["n_pairs"] >= MIN_PAIRS)
    limit = symmetric_limit([monthly])
    panels = [
        (monthly.sel(month=month), calendar.month_name[month], BIAS_CMAP, -limit, limit, "SKT - LST [K]")
        for month in range(1, 13)
    ]
    map_figure(
        panels, f"Monthly mean clear-sky bias, {YEAR}", fig_dir / "bias_monthly_maps.pdf", 4, (16, 12), shrink=0.5
    )


def figure_bias_diurnal(cubes, fig_dir):
    hourly = cubes["hourly"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cycle = hourly["bias_mean"].where(hourly["n_pairs"] > 0).mean(("lat", "lon"))  # (month, hour)
        cells = (hourly["n_pairs"] > 0).sum(("lat", "lon"))
    limit = symmetric_limit([cycle], quantile=1.0)
    fig, (ax_heat, ax_lines) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    image = ax_heat.imshow(cycle.values, cmap=BIAS_CMAP, vmin=-limit, vmax=limit, aspect="auto", origin="upper")
    ax_heat.set_yticks(range(12))
    ax_heat.set_yticklabels([calendar.month_abbr[month] for month in range(1, 13)])
    ax_heat.set_xticks(range(0, 24, 3))
    ax_heat.set_xlabel("local solar hour")
    ax_heat.set_title("continent mean bias by month and hour")
    colorbar = fig.colorbar(image, ax=ax_heat, shrink=0.8)
    colorbar.set_label("SKT - LST [K]", color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    colors = plt.get_cmap("twilight")(np.linspace(0.05, 0.95, 12))
    for month in range(1, 13):
        ax_lines.plot(
            range(24), cycle.sel(month=month), color=colors[month - 1], lw=1.4, label=calendar.month_abbr[month]
        )
    ax_lines.axhline(0, color="black", lw=0.6)
    ax_lines.set_xlabel("local solar hour")
    ax_lines.set_ylabel("SKT - LST [K]")
    ax_lines.set_xticks(range(0, 24, 3))
    ax_lines.set_title(f"diurnal cycle of the bias per month (mean over {int(cells.max())} cells at most)")
    ax_lines.legend(ncol=4, fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax_lines.spines[side].set_visible(False)
    fig.suptitle(f"Clear-sky bias of ERA5-Land SKT against SEVIRI LST, {YEAR}", fontsize=13)
    path = fig_dir / "bias_diurnal_cycle.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_bias_day_night(cubes, fig_dir):
    day = weighted_hour_mean(cubes["hourly"], DAY_HOURS)
    night = weighted_hour_mean(cubes["hourly"], NIGHT_HOURS)
    limit = symmetric_limit([day, night])
    day_label = f"daytime peak ({DAY_HOURS[0]:02d}-{DAY_HOURS[-1]:02d} local)"
    night_label = f"night trough ({NIGHT_HOURS[0]:02d}-{NIGHT_HOURS[-1]:02d} local)"
    panels = [
        (day, day_label, BIAS_CMAP, -limit, limit, "SKT - LST [K]"),
        (night, night_label, BIAS_CMAP, -limit, limit, "SKT - LST [K]"),
    ]
    map_figure(
        panels,
        f"Mean clear-sky bias at the daytime peak and at night, {YEAR}",
        fig_dir / "bias_day_night_maps.pdf",
        2,
        (13, 6),
    )


def figure_bias_extrema(cubes, fig_dir):
    daily, monthly = cubes["daily"], cubes["monthly"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        max_map = daily["bias_max"].mean("day")
        min_map = daily["bias_min"].mean("day")
        max_series = monthly["bias_max_mean"].mean(("lat", "lon"))
        min_series = monthly["bias_min_mean"].mean(("lat", "lon"))
    limit = symmetric_limit([max_map, min_map])
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[3, 1.2])
    panels = [(max_map, "bias of the daily maximum"), (min_map, "bias of the daily minimum")]
    for column, (field, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column], projection=ccrs.PlateCarree())
        image = draw_map(ax, field, BIAS_CMAP, -limit, limit)
        ax.set_title(title, fontsize=11, color=LABEL_COLOR)
    colorbar = fig.colorbar(image, ax=fig.axes, shrink=0.5, pad=0.02)
    colorbar.set_label("SKT - LST [K]", color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    ax = fig.add_subplot(grid[1, :])
    ax.plot(monthly["month"], max_series, marker="o", color="#b3412f", label="daily maximum")
    ax.plot(monthly["month"], min_series, marker="o", color="#2f5fb3", label="daily minimum")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[month] for month in range(1, 13)])
    ax.set_ylabel("SKT - LST [K]")
    ax.set_title("continent mean of the per-cell monthly means")
    ax.legend(frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(f"Bias of the daily extrema over the paired clear-sky hours, {YEAR}", fontsize=13)
    path = fig_dir / "bias_extrema_maps.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_amplitude_error(cubes, fig_dir):
    """The error of the diurnal amplitude: per cell the annual mean of the daily range of
    SKT minus the daily range of LST over the paired clear-sky hours (the bias of the daily
    maximum minus the bias of the daily minimum), with the continent-mean monthly series.
    The variance error of a diurnal signal in one number: negative = a damped daily cycle."""
    daily, monthly = cubes["daily"], cubes["monthly"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        amplitude_map = (daily["bias_max"] - daily["bias_min"]).mean("day")
        amplitude_series = (monthly["bias_max_mean"] - monthly["bias_min_mean"]).mean(("lat", "lon"))
    limit = symmetric_limit([amplitude_map])
    fig = plt.figure(figsize=(8, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1.2])
    ax = fig.add_subplot(grid[0], projection=ccrs.PlateCarree())
    image = draw_map(ax, amplitude_map, BIAS_CMAP, -limit, limit)
    ax.set_title("diurnal amplitude error: daily range of SKT $-$ daily range of LST", fontsize=11, color=LABEL_COLOR)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.6, pad=0.02)
    colorbar.set_label("SKT - LST daily range [K]", color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    ax = fig.add_subplot(grid[1])
    ax.plot(monthly["month"], amplitude_series, marker="o", color="#b3412f")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[month] for month in range(1, 13)])
    ax.set_ylabel("SKT - LST daily range [K]")
    ax.set_title("continent mean of the per-cell monthly means")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.suptitle(f"Diurnal amplitude error over the paired clear-sky hours, {YEAR}", fontsize=13)
    path = fig_dir / "amplitude_error_map.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_bias_timeseries(cubes, fig_dir):
    hourly, daily, monthly = cubes["hourly"], cubes["daily"], cubes["monthly"]
    fig, (ax_bias, ax_count) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_bias.plot(hourly["time"], hourly["continent_bias"], color="#bbbbbb", lw=0.4, label="hourly")
    ax_bias.plot(daily["day"], daily["continent_bias_mean"], color="#2f5fb3", lw=1.2, label="daily")
    month_starts = pd.to_datetime([f"{YEAR}-{month:02d}-01" for month in range(1, 13)] + [f"{YEAR + 1}-01-01"])
    ax_bias.stairs(
        monthly["continent_bias"].values, month_starts, color="#b3412f", lw=2, baseline=None, label="monthly"
    )
    ax_bias.axhline(0, color="black", lw=0.6)
    ax_bias.set_ylabel("SKT - LST [K]")
    ax_bias.set_title("continent mean clear-sky bias at three temporal scales")
    ax_bias.legend(frameon=False, loc="upper right")
    ax_count.plot(
        hourly["time"], hourly["continent_n_cells"], color="#bbbbbb", lw=0.4, label="cells with a pair (hourly)"
    )
    ax_count.plot(daily["day"], daily["continent_n_cells"], color="#2f5fb3", lw=1.2, label="cells with pairs (daily)")
    ax_count.set_ylabel("cells")
    ax_count.legend(frameon=False, loc="upper right")
    for ax in (ax_bias, ax_count):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"ERA5-Land SKT minus SEVIRI LST over the African continent, {YEAR}", fontsize=13)
    path = fig_dir / "bias_timeseries.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def figure_spearman_maps(cubes, fig_dir):
    rho = cubes["spearman"]["rho"]
    obs_name, model_name = pair_names(cubes)
    series, window_days, _ = series_label(cubes)
    window_month = rho["window"].dt.month
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fields = [("annual mean", rho.mean("window"))]
        for season, months in SEASONS.items():
            fields.append((season, rho.sel(window=window_month.isin(months)).mean("window")))
    has_negative = any(float(field.min()) < 0 for _, field in fields)
    cmap, vmin = ("RdBu_r", -1) if has_negative else ("viridis", 0)
    panels = [(field, title, cmap, vmin, 1, "Spearman rho") for title, field in fields]
    map_figure(
        panels,
        f"Spearman rank correlation of {series} {model_name} and {obs_name} in {window_days}-day windows, {YEAR}",
        fig_dir / "spearman_maps.pdf",
        3,
        (15, 9),
        shrink=0.5,
    )


def figure_spearman_timeseries(cubes, fig_dir):
    spearman = cubes["spearman"]
    series, window_days, min_pairs = series_label(cubes)
    rho = spearman["rho"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        quartiles = rho.quantile([0.25, 0.75], dim=("lat", "lon"), skipna=True)
    fig, (ax_rho, ax_count) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_rho.fill_between(
        rho["window"],
        quartiles.sel(quantile=0.25),
        quartiles.sel(quantile=0.75),
        color="#2f5fb3",
        alpha=0.2,
        label="interquartile range across cells",
    )
    ax_rho.plot(rho["window"], spearman["continent_rho"], color="#2f5fb3", lw=1.5, label="mean over cells")
    ax_rho.set_ylabel("Spearman rho")
    lower = max(-1.0, float(np.floor(quartiles.min() * 10) / 10))  # data-driven limits
    upper = min(1.0, float(np.ceil(quartiles.max() * 10) / 10))
    ax_rho.set_ylim(lower, upper)
    ax_rho.axhline(0, color="black", lw=0.6)
    ax_rho.legend(frameon=False, loc="lower right")
    obs_name, model_name = pair_names(cubes)
    ax_rho.set_title(f"rolling {window_days}-day Spearman correlation of {series} {model_name} and {obs_name}")
    ax_count.plot(rho["window"], spearman["continent_n_cells"], color="#2f5fb3", lw=1.2)
    ax_count.set_ylabel(f"cells with >= {min_pairs} pairs")
    for ax in (ax_rho, ax_count):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{model_name} versus {obs_name} over the African continent, {YEAR}", fontsize=13)
    path = fig_dir / "spearman_timeseries.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_pairs_density(cubes, fig_dir):
    sample = cubes["pairs_sample"]
    pair = PAIRS[sample.attrs["pair"]]
    obs_name, obs_unit = LABELS[pair["obs"]]
    model_name, model_unit = LABELS[pair["model"]]
    obs, model = sample["obs"].values, sample["model"].values
    is_hourly = pair["mode"] == "hourly"
    if is_hourly:
        is_day = (sample["local_hour"].values >= 6) & (sample["local_hour"].values < 18)
        panels = [(is_day, "day (06-18 local)"), (~is_day, "night (18-06 local)")]
    else:
        panels = [(np.ones(len(obs), dtype=bool), f"all days of {YEAR}")]
        obs_name = f"daytime {obs_name} ({DAY_HOURS[0]:02d}-{DAY_HOURS[-1]:02d} local)"
        model_name = f"daily mean {model_name}"
    if is_hourly:  # same quantity: shared axes and a 1:1 line
        low = float(np.floor(min(obs.min(), model.min()) / 10) * 10)
        high = float(np.ceil(max(obs.max(), model.max()) / 10) * 10)
        extent = (low, high, low, high)
    else:
        extent = (*np.quantile(obs, [0.001, 0.999]), *np.quantile(model, [0.001, 0.999]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 6), constrained_layout=True, squeeze=False)
    for ax, (select, title) in zip(axes[0], panels):
        x, y = obs[select], model[select]
        image = ax.hexbin(x, y, gridsize=80, bins="log", extent=extent, cmap="magma_r", mincnt=1)
        if is_hourly:
            ax.plot([low, high], [low, high], color="black", lw=0.8, ls="--")
            bias = float(np.mean(y - x))
            rmse = float(np.sqrt(np.mean((y - x) ** 2)))
            stats = f"n = {len(x):,}\nbias {bias:+.2f} {obs_unit}\nRMSE {rmse:.2f} {obs_unit}"
            ax.set_aspect("equal")
        else:
            rho = spearmanr(x, y).statistic
            stats = f"n = {len(x):,}\nSpearman rho {rho:+.2f}"
        ax.text(0.04, 0.94, stats, transform=ax.transAxes, va="top")
        ax.set_xlabel(f"{obs_name} [{obs_unit}]")
        ax.set_ylabel(f"{model_name} [{model_unit}]")
        ax.set_title(title)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
        colorbar.set_label("pairs (log)", color=LABEL_COLOR)
        colorbar.outline.set_visible(False)
    kind = "clear-sky hourly" if is_hourly else "daily"
    fig.suptitle(f"Random sample of the {kind} pairs, {YEAR}", fontsize=13)
    path = fig_dir / "pairs_density.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_frois_ncovar(cubes, fig_dir):
    """Ncovar map and the per-month share of cells with a significant coupling."""
    frois = cubes["frois"]
    obs_name, model_name = pair_names(cubes)
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_map = fig.add_subplot(grid[0], projection=ccrs.PlateCarree())
    image = draw_map(ax_map, frois["n_covar"], "YlOrRd", 0, 12)
    ax_map.set_title("Ncovar = sum of the monthly weights", fontsize=11, color=LABEL_COLOR)
    colorbar = fig.colorbar(image, ax=ax_map, shrink=0.7, pad=0.02)
    colorbar.set_label("months with a significant coupling", color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    ax_bar = fig.add_subplot(grid[1])
    in_continent = frois["n_covar"].notnull()
    weight = frois["weight"].where(in_continent)
    rho = frois["rho"].where(in_continent)
    negative = (weight * (rho < 0)).mean(("lat", "lon"))
    positive = (weight * (rho > 0)).mean(("lat", "lon"))
    month_labels = [calendar.month_abbr[month] for month in range(1, 13)]
    ax_bar.bar(month_labels, negative, color="#2f5fb3", label="negative rho")
    ax_bar.bar(month_labels, positive, bottom=negative, color="#b3412f", label="positive rho")
    ax_bar.set_ylabel("mean weight over continent cells")
    ax_bar.set_title("weight per calendar month, split by the sign of rho")
    ax_bar.legend(frameon=False)
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)
    years = f"{FROIS_YEARS[0]}-{FROIS_YEARS[1]}"
    fig.suptitle(f"Co-variability of {obs_name} and {model_name} (frois Ncovar), {years}", fontsize=13)
    path = fig_dir / "frois_ncovar.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def figure_frois_monthly(cubes, fig_dir):
    """Interannual Spearman rho per calendar month, grey where the weight is zero."""
    frois = cubes["frois"]
    obs_name, model_name = pair_names(cubes)
    rho = frois["rho"].where(frois["weight"] > 0)
    panels = [
        (rho.sel(month=month), calendar.month_name[month], "RdBu_r", -1, 1, "Spearman rho") for month in range(1, 13)
    ]
    years = f"{FROIS_YEARS[0]}-{FROIS_YEARS[1]}"
    title = f"Interannual Spearman rho of monthly {obs_name} and {model_name}, {years} (weight > 0 only)"
    map_figure(panels, title, fig_dir / "frois_monthly_rho.pdf", 4, (16, 12), shrink=0.5)


BIAS_FIGURES = [
    figure_bias_annual,
    figure_bias_monthly,
    figure_bias_diurnal,
    figure_bias_day_night,
    figure_bias_extrema,
    figure_amplitude_error,
    figure_bias_timeseries,
]
CORRELATION_FIGURES = [figure_spearman_maps, figure_spearman_timeseries, figure_pairs_density]
FROIS_FIGURES = [figure_frois_ncovar, figure_frois_monthly]


def plot(args):
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    cubes = open_cubes(args.cube_dir)
    pair = PAIRS[cubes["spearman"].attrs["pair"]]
    if pair["mode"] == "hourly":
        figures = BIAS_FIGURES + CORRELATION_FIGURES
    else:
        figures = CORRELATION_FIGURES + FROIS_FIGURES
    for figure in figures:
        figure(cubes, args.fig_dir)


# %% ============================== ANIMATION ==========================================
GLOBAL_WORKER = {}  # opened source and day range of the global band workers (set by init_global_worker)


def init_global_worker(source, days):
    GLOBAL_WORKER["opened"] = open_source(source)
    GLOBAL_WORKER["days"] = days


def read_global_band(band):
    """(band rows, cols, days, 24) of whole latitude rows: one contiguous read, the cells
    being row-major on the grid."""
    row_start, row_stop = band
    opened = GLOBAL_WORKER["opened"]
    n_cols = opened["grid"].get_shape()[1]
    block = read_hourly(opened, row_start * n_cols, row_stop * n_cols, *GLOBAL_WORKER["days"])
    return block.reshape(row_stop - row_start, n_cols, -1, 24)


def global_hourly(source, days, workers):
    """(rows, cols, days, 24) of one source on its whole grid, read in latitude bands by
    forked workers (every band decompresses its own store chunks, so the bands parallelise)."""
    n_rows = store_grid(source["modality"]).get_shape()[0]
    bands = []
    for band_start in range(0, n_rows, GLOBAL_BAND_ROWS):
        bands.append((band_start, min(band_start + GLOBAL_BAND_ROWS, n_rows)))
    context = multiprocessing.get_context("fork")
    with context.Pool(workers, initializer=init_global_worker, initargs=(source, days)) as pool:
        blocks = pool.map(read_global_band, bands)
    return np.concatenate(blocks)


def hourly_window(source, start, n_days, window, workers):
    """Hourly values of one SOURCES entry: (hours, rows, cols) float32, the UTC hour axis
    and the mask of the drawn cells. window 'africa' is the ERA5-Land analysis window,
    aligned as for the statistics (load_band: SEVIRI LST as the 3 x 3 clear-sky mean, NaN
    below MIN_CLEAR), NaN outside the continent polygon, which is the mask; 'global' is
    the source's whole grid, the mask being the cells finite at any hour (land)."""
    day_start = T1H.date_to_index(start) // 24
    days = (day_start, day_start + n_days)
    if window == "global":
        block = global_hourly(source, days, workers)
        n_rows, n_cols = block.shape[:2]
        stack = block.reshape(n_rows, n_cols, n_days * 24).transpose(2, 0, 1)
        mask = np.isfinite(stack).any(axis=0)
    else:
        grid = store_grid(SOURCES["skt"]["modality"])
        (row_start, row_stop), (col_start, col_stop) = target_window(grid)
        lats = grid.y[row_start:row_stop]
        lons = grid.x[col_start:col_stop]
        mask = africa_mask(lats, lons)
        block = load_band(open_source(source), lats, lons, days, MIN_CLEAR)  # (rows, cols, days, 24)
        stack = block.reshape(len(lats), len(lons), n_days * 24).transpose(2, 0, 1)
        stack[:, ~mask] = np.nan
    hours = pd.date_range(start, periods=n_days * 24, freq="h")
    return stack, hours, mask


def value_limits(values, lo=1, hi=99):
    """Colour limits from the finite values (the split-window convention)."""
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def time_stamp(hour):
    return f"{hour:%Y-%m-%dT%H:%M} UTC"


def frame_figure(field, mask, cmap, vmin, vmax, title, label):
    """One frame in the layout of the split-window walkthrough's gifs (sw_algorithm
    gif_single): the bare map (no frame or axes) with the time stamp above and a fixed
    colorbar at the right, the figure as wide as the field's aspect asks (square for the
    continent, 2:1 for the globe). Missing cells inside the mask (clouds for LST, water
    for SKT) grey, everything outside it white."""
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(MISSING_COLOR)
    n_rows, n_cols = field.shape
    aspect = max(1.0, n_cols / n_rows)
    fig = plt.figure(figsize=(ANIMATION_INCHES * aspect, ANIMATION_INCHES))
    ax = fig.add_axes([0.01, 0.01, 0.80, 0.90])
    cax = fig.add_axes([0.84, 0.12, 0.035 / aspect, 0.68])
    image = ax.imshow(
        field, cmap=colormap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal", interpolation="nearest"
    )
    outside = np.where(mask, np.nan, 1.0)
    ax.imshow(
        outside, cmap=ListedColormap(["white"]), vmin=0, vmax=1, origin="upper", aspect="equal", interpolation="nearest"
    )
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.colorbar(image, cax=cax, label=label)
    return fig


def frame_image(fig, dpi=100):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_gif(images, path, fps):
    """One global palette for every frame (quantised from a thinned, downscaled sample of
    them), so the fixed elements do not flicker between frames."""
    sample = []
    for image in images[::6]:
        small = image.resize((image.width // 3, image.height // 3))
        sample.append(np.asarray(small))
    master = Image.fromarray(np.concatenate(sample, axis=0)).quantize(colors=256, method=Image.MEDIANCUT)
    frames = []
    for image in images:
        frames.append(image.quantize(palette=master, dither=Image.Dither.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, disposal=2)
    print(f"wrote {path} ({len(frames)} frames)")


def animate(args):
    """Hourly SKT on the globe and SEVIRI LST over the continent for a few days, one gif
    each in the split-window walkthrough's layout plus one hour of each as a pdf. Each
    field keeps its own colour range (its 1-99 percentiles over every frame)."""
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    tag = None
    for name in ANIMATION_SOURCES:
        window = ANIMATION_WINDOWS[name]
        stack, hours, mask = hourly_window(SOURCES[name], args.start, args.days, window, args.workers)
        vmin, vmax = value_limits(stack)
        print(
            f"{name} ({window}): {len(hours)} hourly frames from {hours[0]} to {hours[-1]}, "
            f"colour range {vmin:.1f}-{vmax:.1f} K",
            flush=True,
        )
        tag = f"{hours[0]:%Y%m%d}_{args.days}d"
        label = f"{LABELS[name][0]} [{LABELS[name][1]}]"
        images = []
        for frame, hour in zip(stack, hours):
            fig = frame_figure(frame, mask, ANIMATION_CMAP, vmin, vmax, time_stamp(hour), label)
            images.append(frame_image(fig))
        save_gif(images, args.fig_dir / f"{name}_hourly_{tag}.gif", args.fps)

        pdf_hour = hours[args.pdf_hour]
        fig = frame_figure(stack[args.pdf_hour], mask, ANIMATION_CMAP, vmin, vmax, time_stamp(pdf_hour), label)
        path = args.fig_dir / f"{name}_hourly_{tag}_{pdf_hour:%Y%m%dT%H}.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}", flush=True)


# %% ============================== CLI ================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compute_parser = subparsers.add_parser("compute", help="pairs, statistics and the NetCDF cubes")
    compute_parser.add_argument("--pair", default=PAIR, choices=sorted(PAIRS))
    compute_parser.add_argument("--year", type=int, default=YEAR)
    compute_parser.add_argument("--cube_dir", type=Path, default=CUBE_DIR)
    compute_parser.add_argument("--workers", type=int, default=8)
    compute_parser.add_argument("--tcc_max", type=float, default=TCC_MAX)
    compute_parser.add_argument("--min_clear", type=int, default=MIN_CLEAR)
    compute_parser.add_argument(
        "--min_years", type=int, default=FROIS_MIN_YEARS, help="daily mode: frois minimum years"
    )
    compute_parser.add_argument(
        "--rows", type=int, nargs=2, metavar=("START", "STOP"), default=None, help="window rows to process (smoke test)"
    )
    compute_parser.set_defaults(run=compute)
    plot_parser = subparsers.add_parser("plot", help="the nine PDFs from the cubes")
    plot_parser.add_argument("--cube_dir", type=Path, default=CUBE_DIR)
    plot_parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    plot_parser.set_defaults(run=plot)
    animate_parser = subparsers.add_parser("animate", help="bare hourly SKT gif of the continent + one pdf frame")
    animate_parser.add_argument("--start", default=ANIMATION_START, help="first day shown, YYYY-MM-DD (UTC)")
    animate_parser.add_argument("--days", type=int, default=ANIMATION_DAYS)
    animate_parser.add_argument("--fps", type=int, default=ANIMATION_FPS, help="frames (hours) per second")
    animate_parser.add_argument(
        "--pdf_hour", type=int, default=ANIMATION_PDF_HOUR, help="frame index (hours from the start) of the pdf"
    )
    animate_parser.add_argument("--fig_dir", type=Path, default=FIG_DIR / "skt")
    animate_parser.add_argument("--workers", type=int, default=ANIMATION_WORKERS, help="forked global band readers")
    animate_parser.set_defaults(run=animate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
