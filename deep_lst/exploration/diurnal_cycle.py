# Exploratory analysis of the diurnal LST cycle at the Dahra evaluation site.
#
# Descriptors computed for dry-season test periods:
#   - mean diurnal composite (24h mean over 14 days)
#   - diurnal NaN fraction  (cloud/QC mask coverage per day × hour)
#   - diurnal temperature range (DTR)
#   - solar time of daily maximum and minimum
#   - morning warming rate and evening cooling rate

import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import xarray as xr

from deep_lst.exploration.plots import plot_cloudiness_heatmap, plot_cloudiness_polar, plot_diurnal_cycle
from deep_lst.utils import ExperimentPaths


def equation_of_time(doy):
    """Equation of time in minutes for day-of-year (Spencer 1971).
    t_solar = t_utc + lon/15 + equation_of_time(doy)/60
    """
    B = 2 * np.pi / 365 * (doy - 1)
    return 229.18 * (
        0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B)
    )


def _circular_mean_std(hours):
    """Circular mean and std for hour-of-day values (0–24 range)."""
    a = 2 * np.pi * np.asarray(hours) / 24
    mean_h = np.arctan2(np.nanmean(np.sin(a)), np.nanmean(np.cos(a))) * 24 / (2 * np.pi) % 24
    R = np.sqrt(np.nanmean(np.sin(a)) ** 2 + np.nanmean(np.cos(a)) ** 2)
    std_h = np.sqrt(-2 * np.log(np.clip(R, 1e-10, 1))) * 24 / (2 * np.pi)
    return mean_h, std_h


def diurnal_descriptors(lst, solar_offset):
    """Descriptors of the diurnal LST cycle from a (N_DAYS, N_HOURS) site-mean array.
    solar_offset: hours; t_solar = t_utc + solar_offset
    Returns dict of {name: (mean, std)} over days.
    """
    dtr = np.nanmax(lst, axis=1) - np.nanmin(lst, axis=1)
    t_max_solar = (np.array([np.nanargmax(r) for r in lst]) + solar_offset) % 24
    t_min_solar = (np.array([np.nanargmin(r) for r in lst]) + solar_offset) % 24

    warm_idx = (np.arange(7, 14) - round(solar_offset)).astype(int) % 24
    cool_idx = (np.arange(14, 22) - round(solar_offset)).astype(int) % 24

    warming = np.array(
        [
            np.polyfit(warm_idx, lst[d, warm_idx], 1)[0] if np.isfinite(lst[d, warm_idx]).all() else np.nan
            for d in range(len(lst))
        ]
    )
    cooling = np.array(
        [
            np.polyfit(cool_idx, lst[d, cool_idx], 1)[0] if np.isfinite(lst[d, cool_idx]).all() else np.nan
            for d in range(len(lst))
        ]
    )

    desc = {
        "dtr": (np.nanmean(dtr), np.nanstd(dtr)),
        "t_max_solar": _circular_mean_std(t_max_solar),
        "t_min_solar": _circular_mean_std(t_min_solar),
        "warming_rate": (np.nanmean(warming), np.nanstd(warming)),
        "cooling_rate": (np.nanmean(cooling), np.nanstd(cooling)),
    }

    print(f"  DTR:          {desc['dtr'][0]:.1f} ± {desc['dtr'][1]:.1f} K")
    print(f"  T_max hour:   {desc['t_max_solar'][0]:.1f} ± {desc['t_max_solar'][1]:.1f} solar")
    print(f"  T_min hour:   {desc['t_min_solar'][0]:.1f} ± {desc['t_min_solar'][1]:.1f} solar")
    print(f"  Warming rate: {desc['warming_rate'][0]:.2f} ± {desc['warming_rate'][1]:.2f} K/h  (solar 07–13h)")
    print(f"  Cooling rate: {desc['cooling_rate'][0]:.2f} ± {desc['cooling_rate'][1]:.2f} K/h  (solar 14–21h)")

    return desc


PATHS = ExperimentPaths()

EVAL_SITES = ["dahra", "gobabeb", "yangambi", "bouamir", "west_africa", "east_africa", "johannesburg", "antananarivo"]
PERIOD_NAMES = [f"test_period_{i:02d}" for i in range(25)]
N_DAYS, N_HOURS = 14, 24  # each test period is exactly 14 days × 24 h

# COLORS = [plt.cm.Blues(x) for x in np.linspace(0.35, 0.9, len(PERIOD_NAMES))]

# Load shared time axis (identical across all eval sites)
times = xr.open_zarr(PATHS.sampled_store_path("seviri", "eval_dahra")).time_idx.values

# Dahra lat/lon center — used later for solar time offset in descriptor computation
_n = 3201
_lons_grid = np.linspace(-80, 80, _n)
_lats_grid = np.linspace(80, -80, _n)
_mapping = np.load(PATHS.eval_indices_path("seviri", "dahra"))["mapping"]
_center_lat = _lats_grid[_mapping // _n].mean()
_center_lon = _lons_grid[_mapping % _n].mean()
print(
    f"Dahra eval region — center: {_center_lat:.2f}°N, {_center_lon:.2f}°E  |  solar noon ≈ {12 - _center_lon / 15:.1f} UTC"
)
site_center_lons = {
    site: _lons_grid[np.load(PATHS.eval_indices_path("seviri", site))["mapping"] % _n].mean() for site in EVAL_SITES
}

period_indices = {name: np.load(PATHS.temporal_indices_path(name))["time_indices"] for name in PERIOD_NAMES}

period_labels = {
    name: (f"{np.datetime_as_string(times[idx[0]], unit='D')} – {np.datetime_as_string(times[idx[-1]], unit='D')}")
    for name, idx in period_indices.items()
}


def _load_site(site):
    lst_full = xr.open_zarr(PATHS.sampled_store_path("seviri", f"eval_{site}"))["LST"]
    diurnal_mean, cloudi_fraction = {}, {}
    for name, idx in period_indices.items():
        lst_period = lst_full.isel(time_idx=idx).values  # (n_pixels, 336) — triggers compute
        cloudi_fraction[name] = np.isnan(lst_period).mean(axis=0).reshape(N_DAYS, N_HOURS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diurnal_mean[name] = np.nanmean(lst_period, axis=0).reshape(N_DAYS, N_HOURS)
    print(f"  {site}: loaded {len(PERIOD_NAMES)} periods")
    return site, diurnal_mean, cloudi_fraction


with Pool(min(len(EVAL_SITES), cpu_count())) as pool:
    results = pool.map(_load_site, EVAL_SITES)

diurnal = {site: diurnal_mean for site, diurnal_mean, _ in results}
nan_frac = {site: cloudi_fraction for site, _, cloudi_fraction in results}


save_dir = PATHS.results_root / "exploration" / "diurnal_cycle"


def solar_offset(best_period, center_lon):
    _center_ts = times[period_indices[best_period][len(period_indices[best_period]) // 2]]
    _center_doy = int((_center_ts.astype("datetime64[D]") - _center_ts.astype("datetime64[Y]")).astype(int)) + 1
    offset = center_lon / 15 + equation_of_time(_center_doy) / 60
    print(
        f"  Solar offset: {offset:+.2f} h  (lon {center_lon / 15:+.2f} h, EoT {equation_of_time(_center_doy):+.1f} min)"
    )
    return offset


def site_descriptors(site):
    print(f"\nSite: {site}")
    best_period = min(nan_frac[site], key=lambda n: nan_frac[site][n].mean())
    print(
        f"Selected: {best_period}  ({period_labels[best_period]})  —  NaN fraction: {nan_frac[site][best_period].mean():.1%}"
    )
    offset = solar_offset(best_period, site_center_lons[site])
    desc = diurnal_descriptors(diurnal[site][best_period], offset)
    return desc, offset


descriptors, site_solar_offsets = {}, {}
for site in EVAL_SITES:
    descriptors[site], site_solar_offsets[site] = site_descriptors(site)

# Population-level warming/cooling slopes: mean over all sites × all test periods
# Each entry is the 14-day mean slope for one (site, period) pair.
all_warming, all_cooling = [], []
for site in EVAL_SITES:
    offset_approx = round(site_center_lons[site] / 15)
    warm_idx = (np.arange(7, 14) - offset_approx) % 24
    cool_idx = (np.arange(14, 22) - offset_approx) % 24
    for name in PERIOD_NAMES:
        lst = diurnal[site][name]
        w = [np.polyfit(warm_idx, lst[d, warm_idx], 1)[0] for d in range(N_DAYS) if np.isfinite(lst[d, warm_idx]).all()]
        c = [np.polyfit(cool_idx, lst[d, cool_idx], 1)[0] for d in range(N_DAYS) if np.isfinite(lst[d, cool_idx]).all()]
        if w:
            all_warming.append(np.mean(w))
        if c:
            all_cooling.append(np.mean(c))

pop_slopes = {
    "warming": (np.mean(all_warming), np.std(all_warming)),
    "cooling": (np.mean(all_cooling), np.std(all_cooling)),
}
print(f"\nPopulation warming rate: {pop_slopes['warming'][0]:.2f} ± {pop_slopes['warming'][1]:.2f} K/h  (solar 07–13h)")
print(f"Population cooling rate: {pop_slopes['cooling'][0]:.2f} ± {pop_slopes['cooling'][1]:.2f} K/h  (solar 14–21h)")

plot_diurnal_cycle(
    diurnal,
    period_labels,
    save_dir / "diurnal_all_sites.pdf",
    nan_frac=nan_frac,
    descriptors=descriptors,
    solar_offsets=site_solar_offsets,
    pop_slopes=pop_slopes,
)
plot_cloudiness_polar(nan_frac, period_labels, save_dir / "cloudiness_polar.pdf")
plot_cloudiness_heatmap(nan_frac, period_labels, save_dir / "cloudiness_heatmap.pdf")
