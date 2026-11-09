import calendar

import xarray as xr

KG_GROUPS = {
    "kg_tropical": (1, 2, 3),  # Af, Am, Aw
    "kg_arid": (4, 5, 6, 7),  # BWh, BWk, BSh, BSk
    "kg_temperate": tuple(range(8, 17)),  # C climates
}
LC_GROUPS = {
    "lc_forest": (1, 2, 3, 4, 5),
    "lc_savanna": (8, 9),
    "lc_grass_crop": (10, 12, 14),
    "lc_shrub_barren": (6, 7, 16),
}
# Annual-mean FVC classes: 1 = bare (FVC 0), 2-10 = nine equal-count bins of the vegetated
# mtg_new sites (edges in assets/spatial/fvc_class_edges.json, per-site class in the
# selection npz via scripts/assets/fvc_site_classes.py).
FVC_GROUPS = {f"fvc_d{fvc_class}": (fvc_class,) for fvc_class in range(1, 11)}
# Vegetation state at init: the pixel's FVC on the init day relative to its climatological
# annual range (0 = at its minimum, 1 = at its maximum; cubes.attach_init_state), in ten
# fixed 0.1-wide bins. Fixed rather than quantile bins because the clipped ends pile up at
# exactly 0 and 1. The top bin is closed above like the top coverage bin.
GREEN_BINS = {f"green_d{k}": ((k - 1) / 10, k / 10) for k in range(1, 11)}
COV_BINS = {
    "cov_lt10": (0.0, 0.10),
    "cov_10_25": (0.10, 0.25),
    "cov_25_50": (0.25, 0.50),
    "cov_50_75": (0.50, 0.75),
    "cov_75_90": (0.75, 0.90),
    "cov_gt90": (0.90, 1.0),
}
# Hours since the last clear-sky observation at or before init, [lo, hi), None = open above:
# the gap the forecast starts from (cubes.attach_init_observation). gap_0 = observed at init.
GAP_BINS = {
    "gap_0": (0, 1),
    "gap_1_6": (1, 7),
    "gap_7_24": (7, 25),
    "gap_25_72": (25, 73),
    "gap_gt72": (73, None),
}
# Display names for the class-group windows; tables and plots fall back to the
# raw window string for anything not listed here.
WINDOW_LABELS = {
    "kg_tropical": "Tropical",
    "kg_arid": "Arid",
    "kg_temperate": "Temperate",
    "lc_forest": "Forest",
    "lc_savanna": "Savanna",
    "lc_grass_crop": "Grass/Crop",
    "lc_shrub_barren": "Shrub/Barren",
    "cov_lt10": "Coverage < 10%",
    "cov_10_25": "Coverage 10-25%",
    "cov_25_50": "Coverage 25-50%",
    "cov_50_75": "Coverage 50-75%",
    "cov_75_90": "Coverage 75-90%",
    "cov_gt90": "Coverage > 90%",
    "init_obs": "Observed at init",
    "init_gap": "Gap at init",
    "init_filled": "Filled at init",
    "gap_0": "Gap 0 h",
    "gap_1_6": "Gap 1-6 h",
    "gap_7_24": "Gap 7-24 h",
    "gap_25_72": "Gap 25-72 h",
    "gap_gt72": "Gap > 72 h",
}
WINDOW_LABELS.update({f"green_d{k}": f"Greenness {(k - 1) * 10}-{k * 10}%" for k in range(1, 11)})
WINDOW_LABELS["fvc_d1"] = "FVC bare"
WINDOW_LABELS.update({f"fvc_d{fvc_class}": f"FVC class {fvc_class}" for fvc_class in range(2, 11)})

DAY_HOURS = (10, 15)
NIGHT_HOURS = (22, 3)
# Local solar hour bins of the diurnal cycle, [lo, hi), for the hour-resolved per-cell verification.
HOUR_BINS = {f"h{lo:02d}_{lo + 4:02d}": (lo, lo + 4) for lo in range(0, 24, 4)}
WINDOW_LABELS.update({name: f"Solar {lo:02d}-{hi:02d} h" for name, (lo, hi) in HOUR_BINS.items()})

MONTH_INITIALS = "JFMAMJJASOND"


def solar_hour(df):
    valid_hour = (df["init"].dt.hour + df["lead"]) % 24
    solar = (valid_hour + df["lon"] / 15) % 24
    return solar


def day_mask(df):
    hour = solar_hour(df)
    return (hour >= DAY_HOURS[0]) & (hour <= DAY_HOURS[1])


def night_mask(df):
    hour = solar_hour(df)
    return (hour >= NIGHT_HOURS[0]) | (hour <= NIGHT_HOURS[1])


def diurnal_mask(df):
    return day_mask(df) | night_mask(df)


def solar_hour_mask(bounds):
    """MASKS-style function keeping samples whose local solar hour is in [lo, hi)."""
    lo, hi = bounds

    def mask(df):
        hour = solar_hour(df)
        return (hour >= lo) & (hour < hi)

    return mask


def class_mask(coord, class_ids):
    """MASKS-style function keeping pixels whose class is in class_ids."""

    def mask(df):
        return df[coord].isin(list(class_ids))

    return mask


def coverage_mask(coord, bounds):
    """MASKS-style function keeping samples whose lookback coverage is in [lo, hi).
    The top bin (hi >= 1) is closed above so full coverage is kept; NaN coverage
    (inits with truncated lookback) fails every bin by construction."""
    lo, hi = bounds
    top_bin = hi >= 1.0

    def mask(df):
        cov = df[coord]
        upper = cov <= hi if top_bin else cov < hi
        return (cov >= lo) & upper

    return mask


def init_observed_mask(df):
    return df["init_observed"]


def init_gap_mask(df):
    return ~df["init_observed"]


def init_filled_mask(df):
    return df["init_filled"]


def range_mask(coord, bounds):
    """MASKS-style function keeping samples whose coord is in [lo, hi); hi None = open above."""
    lo, hi = bounds

    def mask(df):
        values = df[coord]
        inside = values >= lo
        if hi is not None:
            inside = inside & (values < hi)
        return inside

    return mask


def month_mask(months):
    """MASKS-style function keeping samples whose valid time falls in months."""

    def mask(df):
        valid_time = df["init"] + df["lead"].astype("timedelta64[h]")
        return valid_time.dt.month.isin(list(months))

    return mask


def combined_mask(mask_a, mask_b):
    """MASKS-style function keeping samples that pass both masks."""

    def mask(df):
        return mask_a(df) & mask_b(df)

    return mask


def stack_windows(df, windows):
    """Stack masked copies along a 'window' dim; one shared verify pass scores them all."""
    slices = []
    for window in windows:
        mask_fn = MASKS[window]
        slices.append(df if mask_fn is None else df.where(mask_fn(df)))
    stacked = xr.concat(slices, dim="window")
    stacked = stacked.assign_coords(window=windows)
    stacked["lead"].attrs["units"] = "hours"
    return stacked


MASKS = {
    "all": None,
    "day": day_mask,
    "night": night_mask,
    "diurnal": diurnal_mask,
}

for group_name, group_ids in KG_GROUPS.items():
    MASKS[group_name] = class_mask("kg_class", group_ids)
for group_name, group_ids in LC_GROUPS.items():
    MASKS[group_name] = class_mask("lc_class", group_ids)
for group_name, group_ids in FVC_GROUPS.items():
    MASKS[group_name] = class_mask("fvc_class", group_ids)
for group_name in list(KG_GROUPS) + list(LC_GROUPS) + list(FVC_GROUPS):
    for tod_name, tod_mask in (("day", day_mask), ("night", night_mask)):
        MASKS[f"{group_name}_{tod_name}"] = combined_mask(MASKS[group_name], tod_mask)
for bin_name, bounds in HOUR_BINS.items():
    MASKS[bin_name] = solar_hour_mask(bounds)
for bin_name, bounds in COV_BINS.items():
    MASKS[bin_name] = coverage_mask("lookback_coverage", bounds)
MASKS["init_obs"] = init_observed_mask
MASKS["init_gap"] = init_gap_mask
MASKS["init_filled"] = init_filled_mask
for init_name, init_mask in (
    ("init_obs", init_observed_mask),
    ("init_gap", init_gap_mask),
    ("init_filled", init_filled_mask),
):
    for tod_name, tod_mask in (("day", day_mask), ("night", night_mask)):
        MASKS[f"{init_name}_{tod_name}"] = combined_mask(init_mask, tod_mask)
for bin_name, bounds in GAP_BINS.items():
    MASKS[bin_name] = range_mask("hours_since_obs", bounds)
for bin_name, bounds in GREEN_BINS.items():
    MASKS[bin_name] = coverage_mask("relative_greenness", bounds)
    for tod_name, tod_mask in (("day", day_mask), ("night", night_mask)):
        MASKS[f"{bin_name}_{tod_name}"] = combined_mask(MASKS[bin_name], tod_mask)


def register_month_windows(months):
    """Registered by parser."""
    if sorted(months) == list(range(1, 13)):
        return []
    initials = "".join(MONTH_INITIALS[m - 1] for m in months)
    name = initials.lower()
    MASKS[name] = month_mask(months)
    MASKS[f"{name}_day"] = combined_mask(MASKS[name], day_mask)
    MASKS[f"{name}_night"] = combined_mask(MASKS[name], night_mask)
    WINDOW_LABELS[name] = initials
    return [name, f"{name}_day", f"{name}_night"]


# Twelve single-month windows (valid-time month, like register_month_windows) for the
# month-by-month maps (gifs.py). Registered at import under fixed names m01-m12: the
# initials-based names above collide for single months (J = Jan/Jun/Jul). No day/night
# variants -- every window stacks a copy of the forecast cube.
MONTHLY_WINDOWS = [f"m{month:02d}" for month in range(1, 13)]
for name, month in zip(MONTHLY_WINDOWS, range(1, 13)):
    MASKS[name] = month_mask((month,))
    WINDOW_LABELS[name] = calendar.month_name[month]
