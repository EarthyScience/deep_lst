"""Generalised Split-Window (GSW) LST retrieval from SEVIRI radiances.

Pedagogic, step-by-step reconstruction for a conference BACKGROUND slide.
Canonical structure follows Atitar et al. (2009), Eq. 4 -- this is NOT an attempt
to reproduce the operational LSA SAF product, only to show the shape of the algorithm.

The code is split into three clearly separated blocks so the steps are easy to read:
    EQUATIONS  -- the physics/maths only (no I/O, no plotting)
    PROCESSING -- data handling: open / load / regrid / sample (no physics)
    PLOTTING   -- figures and GIFs only (no physics)
The `# %%` STEP sections at the bottom just orchestrate: processing -> equation -> plot.

Run top-to-bottom. Environment: ml_basic.
"""

# %% ============================== CONFIG (paths, constants, tables) ==================
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from pyproj import CRS, Transformer

RAD_ZARR = "/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/Geostationary/HRSEVIRI/HRSEVIRI_channels.zarr"
ERA5_ZARR = "/Net/Groups/BGI/tscratch/mwessel/wegen-stores/era5.zarr"
MODIS_LC = "/Net/Groups/BGI/scratch/shams/weathergenerator/modis_landcover/MCD12C1.A2024001.061.2025216131527.hdf"
MLST_FILE = (
    "/Net/Groups/BGI/data/DataStructureMDI/DATA/Incoming/Geostationary/MLST/"
    "datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MLST/NETCDF/2020/06/15/"
    "NETCDF4_LSASAF_MSG_LST_MSG-Disk_202006151200.nc"
)
FIG_DIR = Path("/Net/Groups/BGI/people/mwessel/deep_lst/results/exploration/sw_algorithm")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Presentation convention: black figure/axes backgrounds, white text/ticks/edges.
plt.rcParams.update(
    {
        "figure.facecolor": "black",
        "savefig.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.edgecolor": "black",
        "text.color": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
    }
)

# SEVIRI geostationary projection -- NOT stored in the zarr, supplied externally.
GEOS_PROJ = dict(proj="geos", lon_0=0, h=35785831, a=6378169, b=6356583.8, sweep="y")
GEOS_CRS = CRS.from_proj4(
    f"+proj=geos +lon_0={GEOS_PROJ['lon_0']} +h={GEOS_PROJ['h']} "
    f"+a={GEOS_PROJ['a']} +b={GEOS_PROJ['b']} +sweep={GEOS_PROJ['sweep']} +units=m +no_defs"
)
LONLAT_TO_GEOS = Transformer.from_crs("EPSG:4326", GEOS_CRS, always_xy=True)
GEOS_TO_LONLAT = Transformer.from_crs(GEOS_CRS, "EPSG:4326", always_xy=True)
R_E = GEOS_PROJ["a"]  # Earth equatorial radius [m]
R_SAT = GEOS_PROJ["a"] + GEOS_PROJ["h"]  # satellite distance from Earth centre [m]

# Channel mapping (canonical SEVIRI band order; zarr attrs only say "band 09"/"band 10").
CH_IR108, CH_IR120 = "channel_9", "channel_10"  # IR10.8, IR12.0

# Planck constants + BT coefficients -- Meteosat-11 (MSG FM-4), EUMETSAT EBBT v2 Sec 7.2.4.
C1, C2 = 1.19104e-5, 1.43877
BT_COEF = {
    CH_IR108: dict(nu_c=931.122, A=0.9983, B=0.6256),
    CH_IR120: dict(nu_c=839.113, A=0.9988, B=0.4002),
}

# IGBP class -> (e11=IR10.8, e12=IR12.0) emissivity. REPRESENTATIVE 10-12 um values
# (NOT the operational product); desert quartz sand has e11<e12 (d_eps<0).
IGBP_E11 = np.array(
    [
        0.991,
        0.985,
        0.986,
        0.985,
        0.985,
        0.985,
        0.972,
        0.962,
        0.978,
        0.975,
        0.970,
        0.983,
        0.975,
        0.965,
        0.975,
        0.985,
        0.945,
    ]
)
IGBP_E12 = np.array(
    [
        0.988,
        0.989,
        0.990,
        0.989,
        0.989,
        0.989,
        0.980,
        0.975,
        0.984,
        0.982,
        0.980,
        0.987,
        0.982,
        0.972,
        0.982,
        0.989,
        0.965,
    ]
)

TARGET_TIME = np.datetime64("2020-06-15T12:00")
GATE_LAT, GATE_LON = 28.55, 23.39  # Libya-4 desert calibration site
# clear-sky screen thresholds (simplified, NOT an operational cloud mask).
T11_COLD, DT_MAX, ANGLE_MAX = 273.0, 6.0, 60.0
# Africa week: chunk-aligned crop (mult. of 232) + strides; one 672-step time chunk.
CROP_Y0, CROP_Y1, CROP_X0, CROP_X1, CROP_SS = 464, 3248, 928, 3480, 4
WEEK_T0, WEEK_T1, WEEK_TS = 646464, 647136, 8  # 2020-06-09..15, 2-hourly -> 84 frames
PROBES = {
    "Libya-4 desert": (28.55, 23.39),
    "Congo rainforest": (0.5, 22.0),
    "Sahel savanna": (13.5, 2.0),
    "Atlantic ocean": (0.0, -10.0),
    "Namib (S-winter)": (-23.0, 18.0),
    "Nile delta crop": (30.5, 31.0),
    "Ethiopia highland": (8.0, 39.0),
}


# %% ============================== EQUATIONS (physics only) ===========================
def radiance_to_bt(R, nu_c, A, B):
    """Effective radiance [mW m-2 sr-1 (cm-1)-1] -> brightness temperature [K]."""
    return (C2 * nu_c / np.log1p(C1 * nu_c**3 / R) - B) / A


def split_window_diff(T11, T12):
    """The split-window signal."""
    return T11 - T12


def tcwv_to_water_vapour(tcwv_kg_m2):
    """ERA5 total column water vapour [kg m-2] -> W [g cm-2]."""
    return tcwv_kg_m2 / 10.0


def landcover_emissivity(lc):
    """IGBP class field -> per-channel emissivity (e11, e12); NaN where class invalid."""
    valid = (lc >= 0) & (lc <= 16)
    idx = np.where(valid, lc, 0)
    e11 = np.where(valid, IGBP_E11[idx], np.nan)
    e12 = np.where(valid, IGBP_E12[idx], np.nan)
    return e11, e12


def view_zenith_angle(lat_deg, lon_deg, r_e=R_E, r_sat=R_SAT):
    """Satellite view-zenith angle [rad] for a sub-satellite point at 0N/0E."""
    latr, lonr = np.radians(lat_deg), np.radians(lon_deg)
    cos_psi = np.cos(latr) * np.cos(lonr)
    d = np.sqrt(r_e**2 + r_sat**2 - 2 * r_e * r_sat * cos_psi)
    return np.arccos(np.clip((r_sat * cos_psi - r_e) / d, -1, 1))


def gsw_lst(T11, dT, eps_mean, d_eps, W, sec2):
    """Generalised split-window LST (Atitar et al. 2009, Eq. 4). sec2 = 1/cos(theta)^2."""
    return (
        T11
        + (1.34 - 0.11 * sec2) * dT
        + (0.29 + 0.08 * sec2) * dT**2
        + (60.67 - 10.01 * sec2) * (1 - eps_mean)
        + (-6.71 + 2.47 * sec2) * W * (1 - eps_mean)
        + (-125.91 + 15.09 * sec2) * d_eps
        + (19.44 - 4.27 * sec2) * W * d_eps
        + (-0.44 + 0.57 * sec2)
    )


def clear_sky_mask(T11, dT, theta_deg):
    """True where the GSW is well-posed: within view angle, not cold cloud / cirrus."""
    return (theta_deg < ANGLE_MAX) & ~((T11 < T11_COLD) | (dT > DT_MAX))


# %% ============================== PROCESSING (data ops, no physics) ==================
def geos_lonlat_grid(xv, yv):
    """geos x/y [m] grids -> per-pixel lon, lat [deg] and an on-disk mask."""
    xx, yy = np.meshgrid(xv, yv)
    lon, lat = GEOS_TO_LONLAT.transform(xx, yy)
    on = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) <= 90)
    return lon, lat, on


def nearest_time_index(times, target):
    return int(np.argmin(np.abs(times - target)))


def lonlat_to_pixel(lat, lon, xv, yv):
    gx, gy = LONLAT_TO_GEOS.transform(lon, lat)
    return int(np.argmin(np.abs(yv - gy))), int(np.argmin(np.abs(xv - gx)))


def load_disk_slice(da, t_index):
    """Full-disk 2D field at one time; missing_value/off-disk -> NaN."""
    a = da.isel(time=t_index).compute().values
    return np.where(a < 1e30, a, np.nan)


def load_crop(da, t_idx, ysl, xsl):
    """Strided (time, y, x) crop into memory; missing -> NaN."""
    a = da.isel(time=t_idx, y=ysl, x=xsl).compute().values
    return np.where(a < 1e30, a, np.nan)


def pixel_timeseries(da, iy, ix, t_slice):
    a = da.isel(time=t_slice, y=iy, x=ix).compute().values
    return np.where(a < 1e30, a, np.nan)


def read_igbp(path):
    """MODIS MCD12C1 majority IGBP land cover (0.05 deg CMG, 3600 x 7200)."""
    from pyhdf.SD import SD, SDC

    return SD(path, SDC.READ).select("Majority_Land_Cover_Type_1")[:].astype(np.int16)


def sample_landcover(igbp, lon, lat, on):
    """Nearest IGBP class at each geos-disk pixel; -1 off-disk."""
    col = np.clip(((np.where(on, lon, 0) + 180.0) / 0.05).astype(int), 0, 7199)
    row = np.clip(((90.0 - np.where(on, lat, 0)) / 0.05).astype(int), 0, 3599)
    return np.where(on, igbp[row, col], -1)


def sample_latlon_grid(field, src_lat, src_lon, q_lat, q_lon, valid, order=1):
    """Sample a REGULAR lat/lon `field` at query lon/lat; NaN where not `valid`.

    Works for any regular grid (ERA5, LSA SAF) -- the spacing/origin are read from
    the 1-D coordinate arrays. order=1 bilinear (smooth fields), order=0 nearest.
    """
    dlat, dlon = src_lat[1] - src_lat[0], src_lon[1] - src_lon[0]
    row = (np.where(valid, q_lat, src_lat[0]) - src_lat[0]) / dlat
    col = (np.where(valid, q_lon, src_lon[0]) - src_lon[0]) / dlon
    out = np.full(np.shape(valid), np.nan, float)
    out[valid] = ndi.map_coordinates(field, [row[valid], col[valid]], order=order, mode="nearest")
    return out


def finite_percentiles(arr, lo=1, hi=99):
    """Colour-scale limits from finite values (a data op, not physics)."""
    v = arr[np.isfinite(arr)]
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# %% ============================== PLOTTING (figures + GIFs only) =====================
def strip_frame(fig):
    """Bare map panels: drop ticks, labels and spines (keep colorbars)."""
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)


def _bare_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_map(field, cmap, vlim, title, cbar_label, fname, extend="neither", rings=None, marker=None):
    """One bare full-disk map panel -> PDF."""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    if rings is not None:
        cs = ax.contour(rings, levels=[30, 45, 60, 75], colors="white", linewidths=0.8)
        ax.clabel(cs, fmt="%d°", fontsize=8, colors="white")
    if marker is not None:
        ax.plot(marker[0], marker[1], "c+", ms=12, mew=2)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, extend=extend, label=cbar_label)
    strip_frame(fig)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120)
    plt.close(fig)
    print(f"saved {fname}")


def plot_pair(panels, suptitle, fname):
    """Two bare map panels side by side -> PDF. panels: (field,cmap,vlim,title,clabel,extend)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, (field, cmap, vlim, ttl, clabel, extend) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, shrink=0.6, extend=extend, label=clabel)
    fig.suptitle(suptitle)
    strip_frame(fig)
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


def plot_timeseries(times, rows, panels, suptitle, fname):
    """Stacked time-series panels, one coloured line per site (axes KEPT)."""
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(rows)))
    fig, axs = plt.subplots(len(panels), 1, figsize=(11, 12), sharex=True)
    for ax, (ylab, key) in zip(axs, panels):
        for col, row in zip(colors, rows):
            ax.plot(times, row[key], color=col, lw=1.0, label=row["name"])
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.2, color="gray")
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axs[0].legend(ncol=4, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(suptitle, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


def plot_scatter_density(x, y, stats_text, title, xlabel, ylabel, fname, lo=270, hi=340):
    """log-density hexbin scatter with 1:1 line (axes KEPT)."""
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(x, y, gridsize=120, bins="log", extent=(lo, hi, lo, hi), cmap="turbo", mincnt=1)
    ax.plot([lo, hi], [lo, hi], color="white", lw=1.0, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(0.04, 0.93, stats_text, transform=ax.transAxes, color="white", fontsize=11, va="top")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="pixel count (log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {fname}")


def _stamp(times, f):
    return np.datetime_as_string(times[f], unit="m") + " UTC"


def _to_img(fig):
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _save_gif(imgs, fname, fps=8):
    # ONE global palette for all frames -> fixed elements (colorbar, title) don't flicker.
    from PIL import Image

    sub = [np.asarray(im.resize((im.width // 3, im.height // 3))) for im in imgs[::6]]
    master = Image.fromarray(np.concatenate(sub, axis=0)).quantize(colors=256, method=Image.MEDIANCUT)
    pf = [im.quantize(palette=master, dither=Image.Dither.NONE) for im in imgs]
    pf[0].save(FIG_DIR / fname, save_all=True, append_images=pf[1:], duration=int(1000 / fps), loop=0, disposal=2)
    print(f"saved {fname} ({len(imgs)} frames)")


def gif_single(stack, times, cmap, lim, label, fname, extend="neither", rings_field=None):
    """Animate a single 2D field over time; fixed-position colorbar."""
    imgs = []
    for f in range(stack.shape[0]):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.01, 0.01, 0.80, 0.90])
        cax = fig.add_axes([0.84, 0.12, 0.035, 0.68])
        im = ax.imshow(stack[f], origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1])
        if rings_field is not None:
            ax.contour(rings_field, levels=[30, 45, 60], colors="white", linewidths=0.5)
        ax.set_title(_stamp(times, f), fontsize=10)
        _bare_ax(ax)
        fig.colorbar(im, cax=cax, label=label, extend=extend)
        imgs.append(_to_img(fig))
    _save_gif(imgs, fname)


def gif_pair(panels, times, fname):
    """Animate two 2D fields side by side; fixed-position colorbars."""
    boxes = [
        ([0.02, 0.04, 0.38, 0.84], [0.405, 0.16, 0.016, 0.60]),
        ([0.55, 0.04, 0.38, 0.84], [0.935, 0.16, 0.016, 0.60]),
    ]
    imgs = []
    for f in range(panels[0][0].shape[0]):
        fig = plt.figure(figsize=(11, 6))
        for (stack, cmap, lim, label, ttl, extend), (abox, cbox) in zip(panels, boxes):
            ax = fig.add_axes(abox)
            cax = fig.add_axes(cbox)
            im = ax.imshow(stack[f], origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(ttl, fontsize=10)
            _bare_ax(ax)
            fig.colorbar(im, cax=cax, label=label, extend=extend)
        fig.suptitle(_stamp(times, f))
        imgs.append(_to_img(fig))
    _save_gif(imgs, fname)


# %% =============================== STEP 0: inventory + radiance gate ==================
print("=== STEP 0: gate ===")
ds = xr.open_zarr(RAD_ZARR, consolidated=True, decode_times=True)
it = nearest_time_index(ds.time.values, TARGET_TIME)
it_pd = nearest_time_index(ds.time.values, np.datetime64("2020-06-15T02:30"))
iy_g, ix_g = lonlat_to_pixel(GATE_LAT, GATE_LON, ds.x.values, ds.y.values)

R_noon = radiance_to_bt(float(ds[CH_IR108].isel(time=it, y=iy_g, x=ix_g)), **BT_COEF[CH_IR108])
R_pd = radiance_to_bt(float(ds[CH_IR108].isel(time=it_pd, y=iy_g, x=ix_g)), **BT_COEF[CH_IR108])
print(
    f"gate BT: noon={R_noon:.2f} K  pre-dawn={R_pd:.2f} K  "
    f"{'PASS' if all(250 < t < 340 for t in (R_noon, R_pd)) else 'FAIL'}"
)

R108 = load_disk_slice(ds[CH_IR108], it)
print(f"IR10.8 radiance: finite {100 * np.isfinite(R108).mean():.1f}%")
plot_map(
    R108,
    "inferno",
    finite_percentiles(R108),
    "Step 0 -- SEVIRI IR10.8 effective radiance (raw)\n2020-06-15 12:00 UTC, full disk",
    "mW m$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$",
    "step0_ir108_radiance.pdf",
    marker=(ix_g, iy_g),
)


# %% =============================== STEP 1: radiance -> brightness temp ================
print("\n=== STEP 1: BT ===")
R120 = load_disk_slice(ds[CH_IR120], it)
rad_lim = finite_percentiles(np.concatenate([R108, R120]))
plot_pair(
    [
        (R108, "inferno", rad_lim, "IR10.8", "radiance", "neither"),
        (R120, "inferno", rad_lim, "IR12.0", "radiance", "neither"),
    ],
    "Step 1 -- SEVIRI effective radiance (Meteosat-11)\n2020-06-15 12:00 UTC, full disk",
    "step1_radiance_r108_r120.pdf",
)

T11 = radiance_to_bt(R108, **BT_COEF[CH_IR108])
T12 = radiance_to_bt(R120, **BT_COEF[CH_IR120])
bt_lim = finite_percentiles(np.concatenate([T11, T12]))
print(f"T11 mean={np.nanmean(T11):.1f} K  T12 mean={np.nanmean(T12):.1f} K")
plot_pair(
    [
        (T11, "inferno", bt_lim, "T11  (IR10.8)", "BT [K]", "neither"),
        (T12, "inferno", bt_lim, "T12  (IR12.0)", "BT [K]", "neither"),
    ],
    "Step 1 -- SEVIRI brightness temperature (Meteosat-11)\n2020-06-15 12:00 UTC, full disk",
    "step1_bt_t11_t12.pdf",
)


# %% =============================== STEP 2: split-window signal dT =====================
print("\n=== STEP 2: dT ===")
dT = split_window_diff(T11, T12)
print(f"dT mean={np.nanmean(dT):.2f} K  percentiles[1,50,99]={np.nanpercentile(dT, [1, 50, 99]).round(2)}")
plot_map(
    dT,
    "cividis",
    (0, 4),
    "Step 2 -- split-window signal  dT = T11 - T12\n2020-06-15 12:00 UTC, full disk",
    "T11 - T12  [K]",
    "step2_split_window_dT.pdf",
    extend="both",
)


# %% =============================== STEP 3: emissivity =================================
print("\n=== STEP 3: emissivity ===")
igbp = read_igbp(MODIS_LC)
lon2d, lat2d, on_disk = geos_lonlat_grid(ds.x.values, ds.y.values)
lc = sample_landcover(igbp, lon2d, lat2d, on_disk)
e11, e12 = landcover_emissivity(lc)
eps_mean, d_eps = (e11 + e12) / 2.0, e11 - e12
print(f"eps_mean mean={np.nanmean(eps_mean):.4f}  d_eps mean={np.nanmean(d_eps):.4f}")
plot_pair(
    [
        (
            eps_mean,
            "viridis",
            (0.95, 0.99),
            r"$\epsilon_{mean} = (\epsilon_{11}+\epsilon_{12})/2$",
            "emissivity",
            "neither",
        ),
        (
            d_eps,
            "RdBu_r",
            (-0.025, 0.025),
            r"$\Delta\epsilon = \epsilon_{11}-\epsilon_{12}$",
            "emissivity difference",
            "neither",
        ),
    ],
    "Step 3 -- channel emissivity from MODIS IGBP land cover\n"
    "(representative per-class values, not the operational product)",
    "step3_emissivity.pdf",
)


# %% =============================== STEP 4: water vapour W + geometry ==================
print("\n=== STEP 4: W + theta ===")
e5 = xr.open_zarr(ERA5_ZARR, consolidated=True)
e5_time = e5.time.values.astype("datetime64[ns]")
it_e5 = nearest_time_index(e5_time, TARGET_TIME)
tcwv = e5.tcwv.isel(time=it_e5).compute().values
W = sample_latlon_grid(
    tcwv_to_water_vapour(tcwv), e5.latitude.values, e5.longitude.values, lat2d, lon2d, on_disk, order=1
)
theta = np.where(on_disk, view_zenith_angle(lat2d, lon2d), np.nan)
sec2 = 1.0 / np.cos(theta) ** 2
theta_deg = np.degrees(theta)
print(f"W mean={np.nanmean(W):.2f} g/cm2  theta max={np.nanmax(theta_deg):.1f} deg")
plot_map(
    W,
    "YlGnBu",
    (0, 6),
    "Step 4 -- column water vapour W (ERA5) + view-zenith rings\n2020-06-15 12:00 UTC, full disk",
    "W = TCWV/10  [g cm$^{-2}$]",
    "step4_water_vapour_theta.pdf",
    extend="max",
    rings=theta_deg,
)


# %% =============================== COMPARISON: dT vs W ================================
print("\n=== COMPARISON: dT vs W ===")
plot_pair(
    [
        (dT, "cividis", (0, 4), "split-window dT = T11 - T12\n(native SEVIRI, 3 km)", "T11 - T12  [K]", "both"),
        (
            W,
            "YlGnBu",
            (0, 6),
            "column water vapour W\n(ERA5, 0.25$\\degree$ ~ 28 km, regridded)",
            "W  [g cm$^{-2}$]",
            "max",
        ),
    ],
    "Comparison -- the clear-sky split-window signal tracks water vapour\n2020-06-15 12:00 UTC, full disk",
    "comparison_dT_vs_W.pdf",
)


# %% =============================== STEP 5: apply GSW Eq. 4 -> LST =====================
print("\n=== STEP 5: GSW LST ===")
LST = gsw_lst(T11, dT, eps_mean, d_eps, W, sec2)
corr = LST - T11
good = np.isfinite(LST) & (theta_deg < ANGLE_MAX)
lst_lim = finite_percentiles(np.where(good, LST, np.nan))
cmax = float(np.nanpercentile(np.abs(corr[good]), 99))
print(f"LST mean={np.nanmean(LST[good]):.2f} K  correction median={np.nanmedian(corr[good]):.2f} K")
plot_pair(
    [
        (LST, "inferno", lst_lim, "LST  (GSW retrieval)", "LST [K]", "neither"),
        (
            corr,
            "RdBu_r",
            (-cmax, cmax),
            "atmospheric + emissivity correction  (LST $-$ T11)",
            "LST $-$ T11 [K]",
            "both",
        ),
    ],
    "Step 5 -- generalised split-window LST (canonical Atitar Eq. 4 structure)\n"
    "2020-06-15 12:00 UTC, full disk (limb >60° not yet masked)",
    "step5_gsw_lst.pdf",
)


# %% =============================== STEP 6: clear-sky + angle masking ==================
print("\n=== STEP 6: masking ===")
clear = clear_sky_mask(T11, dT, theta_deg) & np.isfinite(LST)
LST_masked = np.where(clear, LST, np.nan)
print(f"clear-sky coverage: {100 * clear.sum() / on_disk.sum():.1f}% of disk")
plot_map(
    LST_masked,
    "inferno",
    finite_percentiles(LST_masked),
    "Step 6 -- clear-sky GSW LST  (cloud + view-angle masked)\n2020-06-15 12:00 UTC, full disk",
    "LST [K]",
    "step6_lst_masked.pdf",
)


# %% =============================== STEP 7: Africa-week diurnal GIFs ===================
print("\n=== STEP 7: Africa-week GIFs ===")
ysl, xsl = slice(CROP_Y0, CROP_Y1, CROP_SS), slice(CROP_X0, CROP_X1, CROP_SS)
week_idx = np.arange(WEEK_T0, WEEK_T1, WEEK_TS)
week_times = ds.time.values[week_idx]
lon_c, lat_c, ond_c = lon2d[ysl, xsl], lat2d[ysl, xsl], on_disk[ysl, xsl]
eps_c, deps_c, sec2_c, thetad_c = eps_mean[ysl, xsl], d_eps[ysl, xsl], sec2[ysl, xsl], theta_deg[ysl, xsl]

R108_w = load_crop(ds[CH_IR108], week_idx, ysl, xsl)
R120_w = load_crop(ds[CH_IR120], week_idx, ysl, xsl)
T11_w = radiance_to_bt(R108_w, **BT_COEF[CH_IR108])
T12_w = radiance_to_bt(R120_w, **BT_COEF[CH_IR120])
dT_w = split_window_diff(T11_w, T12_w)

e5_idx = [nearest_time_index(e5_time, t) for t in week_times]
tcwv_w = e5.tcwv.isel(time=e5_idx).compute().values
W_w = np.stack(
    [
        sample_latlon_grid(
            tcwv_to_water_vapour(tcwv_w[f]), e5.latitude.values, e5.longitude.values, lat_c, lon_c, ond_c, order=1
        )
        for f in range(len(week_idx))
    ]
)

LST_w = gsw_lst(T11_w, dT_w, eps_c, deps_c, W_w, sec2_c)
corr_w = LST_w - T11_w
clear_w = clear_sky_mask(T11_w, dT_w, thetad_c) & np.isfinite(LST_w)
LSTm_w = np.where(clear_w, LST_w, np.nan)
print(f"clear-sky coverage over week: {100 * np.nanmean(clear_w):.1f}% of crop")

rad_lim_w = finite_percentiles(np.concatenate([R108_w, R120_w]))
bt_lim_w = finite_percentiles(np.concatenate([T11_w, T12_w]))
lst_lim_w = finite_percentiles(LSTm_w)
corr_cmax = float(np.nanpercentile(np.abs(np.where(clear_w, corr_w, np.nan)), 99))

gif_single(R108_w, week_times, "inferno", rad_lim_w, "IR10.8 radiance", "step7_radiance_ir108.gif")
gif_pair(
    [
        (R108_w, "inferno", rad_lim_w, "radiance", "IR10.8", "neither"),
        (R120_w, "inferno", rad_lim_w, "radiance", "IR12.0", "neither"),
    ],
    week_times,
    "step7_radiance_pair.gif",
)
gif_pair(
    [
        (T11_w, "inferno", bt_lim_w, "BT [K]", "T11 (IR10.8)", "neither"),
        (T12_w, "inferno", bt_lim_w, "BT [K]", "T12 (IR12.0)", "neither"),
    ],
    week_times,
    "step7_bt.gif",
)
gif_single(dT_w, week_times, "cividis", (0, 4), "T11 - T12 [K]", "step7_dT.gif", extend="both")
gif_single(
    W_w, week_times, "YlGnBu", (0, 6), "W [g cm$^{-2}$]", "step7_water_vapour.gif", extend="max", rings_field=thetad_c
)
gif_pair(
    [
        (LST_w, "inferno", lst_lim_w, "LST [K]", "LST (GSW)", "neither"),
        (corr_w, "RdBu_r", (-corr_cmax, corr_cmax), "LST - T11 [K]", "correction", "both"),
    ],
    week_times,
    "step7_lst_correction.gif",
)
gif_single(LSTm_w, week_times, "inferno", lst_lim_w, "LST [K]", "step7_masked_lst.gif")  # hero


# %% =============================== PROBE PIXELS: diurnal time series ==================
print("\n=== PROBE PIXELS ===")
ws_full = slice(WEEK_T0, WEEK_T1)
ptimes = ds.time.values[ws_full]
e5_wk = np.where((e5_time >= ptimes[0]) & (e5_time <= ptimes[-1]))[0]
_ts = ptimes.astype("datetime64[s]").astype(float)
_es = e5_time[e5_wk].astype("datetime64[s]").astype(float)

rows = []
for nm, (la, lo) in PROBES.items():
    iy, ix = lonlat_to_pixel(la, lo, ds.x.values, ds.y.values)
    t11 = radiance_to_bt(pixel_timeseries(ds[CH_IR108], iy, ix, ws_full), **BT_COEF[CH_IR108])
    t12 = radiance_to_bt(pixel_timeseries(ds[CH_IR120], iy, ix, ws_full), **BT_COEF[CH_IR120])
    dt = split_window_diff(t11, t12)
    ila = int(np.argmin(np.abs(e5.latitude.values - la)))
    ilo = int(np.argmin(np.abs(e5.longitude.values - lo)))
    w_hr = tcwv_to_water_vapour(e5.tcwv.isel(time=e5_wk, latitude=ila, longitude=ilo).compute().values)
    w = np.interp(_ts, _es, w_hr)
    lst = gsw_lst(t11, dt, eps_mean[iy, ix], d_eps[iy, ix], w, sec2[iy, ix])
    clr = clear_sky_mask(t11, dt, theta_deg[iy, ix]) & np.isfinite(lst)
    rows.append(dict(name=nm, t11=t11, dt=dt, w=w, lstm=np.where(clr, lst, np.nan)))

plot_timeseries(
    ptimes.astype("datetime64[ns]"),
    rows,
    [
        ("T11 BT [K]\n(raw, cloud-hit)", "t11"),
        ("dT [K]", "dt"),
        ("W [g cm$^{-2}$]", "w"),
        ("clear-sky LST [K]\n(gaps = cloud)", "lstm"),
    ],
    "Probe pixels -- diurnal cycle over the week (2020-06-09 .. 06-15 UTC)",
    "probe_pixel_timeseries.pdf",
)


# %% =============================== STEP 8: validation vs LSA SAF MLST =================
print("\n=== STEP 8: validation ===")
msaf = xr.open_dataset(MLST_FILE)
saf = msaf.LST.isel(time=0).values + 273.15  # Celsius -> K; NaN over cloud/ocean/bad
inside = on_disk & (np.abs(lat2d) <= 80) & (np.abs(lon2d) <= 80)
saf_disk = sample_latlon_grid(saf, msaf.lat.values, msaf.lon.values, lat2d, lon2d, inside, order=0)

both = np.isfinite(LST_masked) & np.isfinite(saf_disk)
ours, theirs = LST_masked[both], saf_disk[both]
diff = ours - theirs
bias, rmse, r = diff.mean(), np.sqrt((diff**2).mean()), np.corrcoef(ours, theirs)[0, 1]
print(f"N={both.sum():,}  bias(ours-SAF)={bias:+.2f} K  RMSE={rmse:.2f} K  R={r:.3f}")
plot_scatter_density(
    theirs,
    ours,
    f"bias {bias:+.2f} K\nRMSE {rmse:.2f} K\nR {r:.3f}",
    f"Step 8 -- our GSW LST vs operational LSA SAF MLST\n2020-06-15 12:00 UTC, clear land pixels (N={both.sum():,})",
    "LSA SAF MLST [K]",
    "our GSW LST [K]",
    "step8_validation_scatter.pdf",
)
