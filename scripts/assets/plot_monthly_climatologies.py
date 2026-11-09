"""Maps of the monthly FVC and LST observational-validity climatologies over Africa.

Reads the twelve NetCDFs of each climatology (work_5 derived, 2016-2024, full seviri grid)
and writes four figures:

    fvc_monthly.png            12 maps of the monthly mean FVC
    lst_validity_monthly.png   12 maps of the monthly share of valid clear-sky LST hours [%]
    annual_mean.png            annual means of both, i.e. the mean over the twelve monthly
                               climatologies (the same convention as fvc_site_classes.py)
    fvc_validity_joint.png     joint distribution of the two annual means over all land pixels
                               in the plotted window (Africa plus Arabia and the Mediterranean
                               rim), with the conditional median of validity per FVC bin

    python scripts/assets/plot_monthly_climatologies.py
"""

import argparse
import calendar
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

FVC_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/FVC_monthly_climatology")
VALIDITY_DIR = Path("/Net/Groups/BGI/work_5/scratch/mwessel/data/derived/LSTvalidity_monthly_climatology")
OUT_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/independent/observational_validity")

# Africa window on the seviri grid (lat runs from north to south in the files).
LAT_SLICE = slice(38.0, -37.0)
LON_SLICE = slice(-20.0, 55.0)

MISSING_COLOR = "#ebebeb"  # ocean and never-observed pixels
LABEL_COLOR = "#52514e"

FVC_STYLE = {"cmap": "Greens", "vmin": 0.0, "vmax": 1.0, "label": "FVC (fraction)"}
VALIDITY_STYLE = {"cmap": "Blues", "vmin": 0.0, "vmax": 100.0, "label": "valid LST hours (%)"}


def load_monthly(directory, file_stem, variable):
    """(12, lat, lon) DataArray of the monthly climatology in the Africa window."""
    monthly = []
    for month in range(1, 13):
        path = directory / f"{file_stem}_{month:02d}.nc"
        field = xr.open_dataset(path)[variable].sel(lat=LAT_SLICE, lon=LON_SLICE)
        monthly.append(field.load())
    stacked = xr.concat(monthly, dim="month")
    stacked["month"] = np.arange(1, 13)
    return stacked


def extent_of(field):
    """[lon_min, lon_max, lat_min, lat_max] of a (lat, lon) field, as cartopy wants it."""
    lon = field["lon"].values
    lat = field["lat"].values
    return [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]


def draw_map(ax, field, style):
    """One field as an image on a cartopy axis with coastlines, no frame or tick labels."""
    cmap = plt.get_cmap(style["cmap"]).copy()
    cmap.set_bad(MISSING_COLOR)
    extent = extent_of(field)
    image = ax.imshow(
        field.values,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=style["vmin"],
        vmax=style["vmax"],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
    )
    ax.coastlines(lw=0.4)
    ax.add_feature(cfeature.BORDERS, lw=0.2, edgecolor="grey")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    return image


def plot_monthly(monthly, style, title, out_path):
    """3 x 4 grid of monthly maps with one shared colorbar."""
    fig, axes = plt.subplots(
        3, 4, figsize=(16, 13), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
    )
    for month, ax in zip(range(1, 13), axes.ravel()):
        image = draw_map(ax, monthly.sel(month=month), style)
        ax.set_title(calendar.month_name[month], fontsize=12, color=LABEL_COLOR)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.5, pad=0.01)
    colorbar.set_label(style["label"], color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_annual(fvc_annual, validity_annual, out_path):
    """Annual mean FVC and annual mean validity side by side."""
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 7), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
    )
    fields = [fvc_annual, validity_annual]
    styles = [FVC_STYLE, VALIDITY_STYLE]
    titles = ["Annual mean FVC", "Annual mean LST observational validity"]
    for ax, field, style, title in zip(axes, fields, styles, titles):
        image = draw_map(ax, field, style)
        ax.set_title(title, fontsize=12, color=LABEL_COLOR)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
        colorbar.set_label(style["label"], color=LABEL_COLOR)
        colorbar.outline.set_visible(False)
    fig.suptitle("Mean of the twelve monthly climatologies, 2016-2024", fontsize=14)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_joint(fvc_annual, validity_annual, out_path):
    """2D histogram of annual validity against annual FVC over the land pixels, with the
    conditional median of validity per FVC bin."""
    fvc = fvc_annual.values.ravel()
    validity = validity_annual.values.ravel()
    both_valid = np.isfinite(fvc) & np.isfinite(validity)
    fvc = fvc[both_valid]
    validity = validity[both_valid]
    correlation = np.corrcoef(fvc, validity)[0, 1]

    fvc_edges = np.linspace(0.0, 1.0, 21)
    bin_centers = 0.5 * (fvc_edges[:-1] + fvc_edges[1:])
    bin_index = np.digitize(fvc, fvc_edges) - 1
    bin_index = np.clip(bin_index, 0, len(bin_centers) - 1)
    medians = np.full(len(bin_centers), np.nan)
    lower = np.full(len(bin_centers), np.nan)
    upper = np.full(len(bin_centers), np.nan)
    for i in range(len(bin_centers)):
        in_bin = validity[bin_index == i]
        if len(in_bin) >= 50:
            lower[i], medians[i], upper[i] = np.quantile(in_bin, [0.1, 0.5, 0.9])

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    counts, _, _, image = ax.hist2d(fvc, validity, bins=[fvc_edges, np.linspace(0.0, 100.0, 41)], cmap="Greys", cmin=1)
    ax.plot(bin_centers, medians, color="#b3412f", lw=2, label="median validity per FVC bin")
    ax.plot(bin_centers, lower, color="#b3412f", lw=1, ls="--", label="10th / 90th percentile")
    ax.plot(bin_centers, upper, color="#b3412f", lw=1, ls="--")
    ax.set_xlabel("annual mean FVC")
    ax.set_ylabel("annual mean valid LST hours (%)")
    ax.set_title(f"land pixels, 20W-55E / 37S-38N (n = {len(fvc):,}), Pearson r = {correlation:.2f}")
    ax.legend(loc="upper right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.set_label("pixels per cell", color=LABEL_COLOR)
    colorbar.outline.set_visible(False)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fvc_monthly = load_monthly(FVC_DIR, "fvc_climatology", "fvc_mean")
    validity_monthly = load_monthly(VALIDITY_DIR, "lst_validity_climatology", "lst_validity")

    fvc_annual = fvc_monthly.mean(dim="month", skipna=True)
    validity_annual = validity_monthly.mean(dim="month", skipna=True)

    plot_monthly(fvc_monthly, FVC_STYLE, "Monthly mean FVC, 2016-2024", args.out_dir / "fvc_monthly.png")
    plot_monthly(
        validity_monthly,
        VALIDITY_STYLE,
        "Monthly LST observational validity, 2016-2024",
        args.out_dir / "lst_validity_monthly.png",
    )
    plot_annual(fvc_annual, validity_annual, args.out_dir / "annual_mean.png")
    plot_joint(fvc_annual, validity_annual, args.out_dir / "fvc_validity_joint.png")


if __name__ == "__main__":
    main()
