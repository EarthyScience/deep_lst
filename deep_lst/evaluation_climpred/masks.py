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
COV_BINS = {
    "cov_lt10": (0.0, 0.10),
    "cov_10_25": (0.10, 0.25),
    "cov_25_50": (0.25, 0.50),
    "cov_50_75": (0.50, 0.75),
    "cov_75_90": (0.75, 0.90),
    "cov_gt90": (0.90, 1.0),
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
}

DAY_HOURS = (10, 15)
NIGHT_HOURS = (22, 3)

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
for group_name in list(KG_GROUPS) + list(LC_GROUPS):
    for tod_name, tod_mask in (("day", day_mask), ("night", night_mask)):
        MASKS[f"{group_name}_{tod_name}"] = combined_mask(MASKS[group_name], tod_mask)
for bin_name, bounds in COV_BINS.items():
    MASKS[bin_name] = coverage_mask("lookback_coverage", bounds)


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
