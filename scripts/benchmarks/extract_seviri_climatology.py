"""Extract hourly and smoothed-hourly SEVIRI LST climatology for an eval region.

Reads the preloaded SEVIRI eval zarr store (seviri_sampled_eval_{region}.zarr)
and computes two climatological baselines from the training years:

  hourly
    For each (DOY, hour) in the test year: mean LST across all training-year
    observations at that exact (DOY, hour).
    Typical sample count per bin: n_train_years (≈ 6).

  smoothed_hourly
    For each (DOY, hour) in the test year: mean LST across all training-year
    observations at that specific *hour* within [DOY − window, DOY + window]
    (circular, wrapping at year boundaries).
    Typical sample count per bin: n_train_years × (2·window + 1) (≈ 126).

Both outputs have dims (valid_time, lat, lon) covering every hour of the test
year so they can be reindexed to any test-period subset at evaluation time,
exactly like the ERA5 baseline.

Saved to
  assets/climatology/hourly/climatology_{eval}.nc
  assets/climatology/smoothed_hourly/climatology_{eval}.nc

Usage
-----
python scripts/benchmarks/extract_seviri_climatology.py --eval dahra
python scripts/benchmarks/extract_seviri_climatology.py --eval dahra \\
    --test_year 2022 --window_days 10

sbatch scripts/benchmarks/extract_seviri_climatology.sh dahra
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from deep_lst.evaluation.spatial import build_local_grid
from deep_lst.utils import ExperimentPaths

PATHS = ExperimentPaths()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--eval", required=True, help="Eval region name, e.g. dahra")
    p.add_argument(
        "--test_year",
        type=int,
        default=2024,
        help="Year to produce the climatology for (default: 2022). All hours of this year are written to the output.",
    )
    p.add_argument(
        "--window_days",
        type=int,
        default=10,
        help="Half-width of the DOY window for smoothed_hourly (default: 10). "
        "A target at (DOY, hour) pools all train observations at that "
        "hour from [DOY − window, DOY + window] (circular).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _build_hourly_clim(
    lst: np.ndarray,  # (n_pixels, n_time)
    doys: np.ndarray,  # (n_time,) — day_of_year 1..366
    hours: np.ndarray,  # (n_time,) — 0..23
) -> np.ndarray:
    """Mean LST at each exact (DOY, hour) across all provided (train) timestamps.

    Returns clim[doy, hour, pixel] with shape (367, 24, n_pixels).
    DOY axis uses 1-based indexing; index 0 is unused.
    """
    n_pixels = lst.shape[0]
    clim = np.full((367, 24, n_pixels), np.nan, dtype=np.float64)

    for doy in np.unique(doys):
        for hour in range(24):
            mask = (doys == doy) & (hours == hour)
            if mask.sum() == 0:
                continue
            clim[doy, hour, :] = np.nanmean(lst[:, mask], axis=1)

    return clim


def _build_smoothed_clim(
    lst: np.ndarray,  # (n_pixels, n_time)
    doys: np.ndarray,  # (n_time,) — 1..366
    hours: np.ndarray,  # (n_time,) — 0..23
    window: int,
    n_doy: int = 366,
) -> np.ndarray:
    """Mean LST at each (DOY, hour) pooling ±window neighbouring DOYs.

    For a target (DOY=d, hour=h): average of all train observations at hour h
    where DOY ∈ [d − window, d + window] (circular, modulo n_doy).

    Returns clim[doy, hour, pixel] with shape (367, 24, n_pixels).
    """
    n_pixels = lst.shape[0]
    clim = np.full((367, 24, n_pixels), np.nan, dtype=np.float64)

    for hour in range(24):
        hour_mask = hours == hour
        if hour_mask.sum() == 0:
            continue
        lst_h = lst[:, hour_mask]  # (n_pixels, n_h)
        doys_h = doys[hour_mask]  # (n_h,)

        for doy in range(1, n_doy + 1):
            # Build the circular DOY window
            offsets = np.arange(-window, window + 1)
            target_doys = ((doy - 1 + offsets) % n_doy) + 1  # 1-based, wrap
            in_window = np.isin(doys_h, target_doys)
            if in_window.sum() == 0:
                continue
            clim[doy, hour, :] = np.nanmean(lst_h[:, in_window], axis=1)

    return clim


def _lookup_clim(
    clim: np.ndarray,  # (367, 24, n_pixels)
    test_timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    """Map test timestamps to climatological values → (n_test, n_pixels)."""
    n_pixels = clim.shape[2]
    result = np.full((len(test_timestamps), n_pixels), np.nan, dtype=np.float32)
    doys = test_timestamps.day_of_year.values
    hours = test_timestamps.hour.values
    for i, (doy, hour) in enumerate(zip(doys, hours)):
        result[i] = clim[doy, hour, :]
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_climatology(
    eval_name: str,
    test_year: int,
    window_days: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    # --- Open preloaded SEVIRI zarr store ---
    store_path = PATHS.sampled_store_path("seviri", f"eval_{eval_name}")
    if not store_path.exists():
        raise FileNotFoundError(
            f"Preloaded SEVIRI store not found: {store_path}\nRun the eval-store preloading step first."
        )
    store = zarr.open(str(store_path), mode="r")

    spatial_idx = store["spatial_idx"][:]
    timestamps = pd.DatetimeIndex(store["time_idx"][:])
    lst = store["LST"][:]  # (n_pixels, n_time) float32

    n_pixels = lst.shape[0]
    print(f"Store  : {n_pixels} pixels × {len(timestamps)} timesteps")
    print(f"Range  : {timestamps[0]} → {timestamps[-1]}")

    train_mask = timestamps.year < test_year
    test_mask = timestamps.year == test_year
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    if n_train == 0:
        raise ValueError(f"No training data found before test year {test_year}.")
    if n_test == 0:
        raise ValueError(f"No data found for test year {test_year}.")

    train_years = f"{timestamps[train_mask][0].year}–{timestamps[train_mask][-1].year}"
    print(f"Train  : {n_train} timesteps  ({train_years})")
    print(f"Test   : {n_test} timesteps  (year {test_year})")

    train_lst = lst[:, train_mask].astype(np.float64)
    train_doys = timestamps[train_mask].day_of_year.values
    train_hours = timestamps[train_mask].hour.values
    test_ts = timestamps[test_mask]

    # --- Hourly climatology ---
    print("Computing hourly climatology…")
    clim_hourly = _build_hourly_clim(train_lst, train_doys, train_hours)

    # --- Smoothed-hourly climatology ---
    print(f"Computing smoothed-hourly climatology (±{window_days} days)…")
    clim_smooth = _build_smoothed_clim(train_lst, train_doys, train_hours, window=window_days)

    # --- Map test timestamps → (n_test, n_pixels) ---
    print("Mapping test timestamps…")
    flat_hourly = _lookup_clim(clim_hourly, test_ts)
    flat_smooth = _lookup_clim(clim_smooth, test_ts)

    # --- Remap pixels → (lat, lon) grid ---
    lats, lons, local_rows, local_cols, grid_shape = build_local_grid(spatial_idx)

    def _to_grid(flat):  # (n_time, n_pix) → (n_time, lat, lon)
        out = np.full((flat.shape[0], *grid_shape), np.nan, dtype=np.float32)
        out[:, local_rows, local_cols] = flat
        return out

    attrs_base = dict(
        eval_region=eval_name,
        test_year=str(test_year),
        train_years=train_years,
        units="K",
    )

    ds_hourly = xr.Dataset(
        {"LST": (["valid_time", "lat", "lon"], _to_grid(flat_hourly))},
        coords={"valid_time": test_ts.values, "lat": lats, "lon": lons},
        attrs={
            **attrs_base,
            "description": "Hourly SEVIRI LST climatology: mean at each (DOY, hour) across train years.",
        },
    )
    ds_smooth = xr.Dataset(
        {"LST": (["valid_time", "lat", "lon"], _to_grid(flat_smooth))},
        coords={"valid_time": test_ts.values, "lat": lats, "lon": lons},
        attrs={
            **attrs_base,
            "description": (
                f"Smoothed-hourly SEVIRI LST climatology: "
                f"mean at each hour within ±{window_days}-day DOY window across train years."
            ),
        },
    )
    return ds_hourly, ds_smooth


def main():
    args = parse_args()

    out_hourly = PATHS.climatology_path(args.eval, "hourly")
    out_smooth = PATHS.climatology_path(args.eval, "smoothed_hourly")

    if out_hourly.exists() and out_smooth.exists() and not args.overwrite:
        print(f"Both files already exist for '{args.eval}'. Pass --overwrite to refresh.")
        return

    ds_hourly, ds_smooth = extract_climatology(args.eval, args.test_year, args.window_days)

    for ds, path, label in [
        (ds_hourly, out_hourly, "hourly"),
        (ds_smooth, out_smooth, "smoothed_hourly"),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(path)
        print(f"Saved {label:20s}: {path}")
        print(f"  valid_time: {ds.valid_time.values[0]} → {ds.valid_time.values[-1]}")
        print(f"  grid: {ds.sizes['lat']}×{ds.sizes['lon']}, {ds.sizes['valid_time']} timesteps")


if __name__ == "__main__":
    main()
