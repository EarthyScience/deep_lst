# Exploratory analysis of the diurnal LST cycle at the Dahra evaluation site.
#
# Descriptors computed for dry-season test periods:
#   - mean diurnal composite (24h mean over 14 days)
#   - diurnal NaN fraction  (cloud/QC mask coverage per day × hour)
#   - diurnal temperature range (DTR)
#   - solar time of daily maximum and minimum
#   - morning warming rate and evening cooling rate

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless cluster script

from deep_lst.dataset.dataset import STORE_EPOCH_OFFSET
from deep_lst.exploration.full_store import SEVIRI_STORE, load_store_columns
from deep_lst.exploration.plots import plot_cloudiness_heatmap, plot_cloudiness_polar, plot_diurnal_cycle
from deep_lst.utils import ExperimentPaths
from deep_lst.utils.spatial import store_grid


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

period_indices = {name: np.load(PATHS.temporal_indices_path(name))["time_indices"] for name in PERIOD_NAMES}


def hours_to_datetime(hours):
    """T1H positions on the 2016-01-01 hourly axis -> datetime64."""
    return np.datetime64("2016-01-01") + hours * np.timedelta64(1, "h")


period_labels = {
    name: (
        f"{np.datetime_as_string(hours_to_datetime(idx[0]), unit='D')} – "
        f"{np.datetime_as_string(hours_to_datetime(idx[-1]), unit='D')}"
    )
    for name, idx in period_indices.items()
}

# Contiguous store-day window covering every test period — the only days read below.
_all_hours = np.concatenate(list(period_indices.values()))
DAY_LO = int((_all_hours.min() + STORE_EPOCH_OFFSET) // 24)
DAY_HI = int((_all_hours.max() + STORE_EPOCH_OFFSET) // 24) + 1

_seviri_centers = store_grid("seviri").centers  # (n_cells, 2) lon/lat


def _load_site(site):
    """Site LST from the full seviri store: eval anchors ARE seviri pixels (rows pass through)."""
    columns, _, _, anchor_ids = load_store_columns(
        "seviri",
        SEVIRI_STORE,
        f"eval_{site}",
        {"data": ["LST"]},
        anchor="seviri",
        day_range=(DAY_LO, DAY_HI),
        num_workers=16,
    )
    lst_full = columns["data"].numpy()[:, :, 0]  # (n_pixels, n_hours_loaded)
    center_lon = float(_seviri_centers[anchor_ids, 0].mean())
    diurnal_mean, cloudi_fraction = {}, {}
    for name, idx in period_indices.items():
        positions = idx + STORE_EPOCH_OFFSET - DAY_LO * 24
        lst_period = lst_full[:, positions]  # (n_pixels, 336)
        cloudi_fraction[name] = np.isnan(lst_period).mean(axis=0).reshape(N_DAYS, N_HOURS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            diurnal_mean[name] = np.nanmean(lst_period, axis=0).reshape(N_DAYS, N_HOURS)
    print(f"  {site}: loaded {len(PERIOD_NAMES)} periods ({len(anchor_ids)} pixels)")
    return center_lon, diurnal_mean, cloudi_fraction


diurnal, nan_frac, site_center_lons = {}, {}, {}
for site in EVAL_SITES:
    site_center_lons[site], diurnal[site], nan_frac[site] = _load_site(site)


save_dir = PATHS.results_root / "exploration" / "diurnal_cycle"


def solar_offset(best_period, center_lon):
    _center_ts = hours_to_datetime(period_indices[best_period][len(period_indices[best_period]) // 2])
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
