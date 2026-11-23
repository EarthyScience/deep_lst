"""Hourly aggregation of the 15-minute Rain over Africa (RoA) rain rates.

The raw product gives a rain rate in mm/h for every 15-minute SEVIRI scan. The store holds one
value per full hour HH: the mean of the rates of the scans that END in the hour before HH, i.e.
in (HH-1:00, HH:00]. A rate in mm/h averaged over one hour equals the rain accumulated in that
hour in mm, so the stored value is both the hourly mean rate and the hourly accumulation. This
is the convention of the accumulated ERA5 fields (tp, ssrd) in era5_product.zarr.

The SEVIRI scan that gives the LST at HH ends at about HH:12 and is therefore not part of the
hour HH: the rain always precedes the LST observation.

Raw values that are not finite or above MAX_VALID_RATE are retrieval artefacts (in 300 random
time steps there was no value between 71 and 10,000 mm/h) and are treated as missing. Tiny
negative values are numerical noise and are set to zero.
"""

import numpy as np

MAX_VALID_RATE = 200.0  # mm/h, limit suggested in the metadata of the raw stores
NEGATIVE_TOLERANCE = 1e-6  # mm/h, values in [-NEGATIVE_TOLERANCE, 0) are set to 0
SLOT_MINUTES = 15  # SEVIRI repeat cycle

RAIN_RATE_SCALE = 0.01  # stored integer * RAIN_RATE_SCALE = mm in the hour
RAIN_RATE_FILL = -32768  # no valid scan in the hour, or cell outside the RoA coverage
N_SCANS_FILL = 0  # the same cells: no scan went into the value
STORE_DTYPE = np.dtype([("rain_rate", "<i2"), ("n_scans", "i1")])
STORE_FILL_VALUE = (RAIN_RATE_FILL, N_SCANS_FILL)


def clean_rates(rates):
    """Separate valid rain rates from retrieval artefacts.

    Args:
        rates: Raw rain rates in mm/h, any shape.

    Returns:
        clean: The rates with tiny negatives set to 0 and invalid values set to 0. float32.
        valid: True where the raw value was a usable rain rate.
    """
    rates = np.asarray(rates)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(rates) & (rates >= -NEGATIVE_TOLERANCE) & (rates <= MAX_VALID_RATE)
    clean = np.zeros(rates.shape, dtype=np.float32)
    clean[valid] = np.maximum(rates[valid], 0.0)
    return clean, valid


def hour_labels(scan_end_times):
    """Full hour at which each scan is provided: the end of the hour in which the scan ends.

    A scan that ends at 13:57:43 belongs to 14:00, a scan that ends at exactly 14:00:00 too,
    and a scan that ends at 14:12:41 belongs to 15:00.

    Args:
        scan_end_times: Observation end time of every scan, datetime64.

    Returns:
        The hour label of every scan, datetime64[h].
    """
    end_times = np.asarray(scan_end_times).astype("datetime64[ns]")
    hour_start = end_times.astype("datetime64[h]")
    ends_on_full_hour = end_times == hour_start
    labels = hour_start + np.timedelta64(1, "h")
    labels[ends_on_full_hour] = hour_start[ends_on_full_hour]
    return labels


def first_scan_per_slot(scan_end_times):
    """Mark the first scan of every 15-minute slot.

    When two satellites cover the same slot, the raw stores hold both scans a few seconds
    apart. Only the first one is used, so that every slot has the same weight in the hour.

    Args:
        scan_end_times: Observation end time of every scan, datetime64, sorted.

    Returns:
        Boolean array, True for the scans to keep.
    """
    end_minutes = np.asarray(scan_end_times).astype("datetime64[m]").astype(np.int64)
    slots = end_minutes // SLOT_MINUTES
    _, first_positions = np.unique(slots, return_index=True)
    keep = np.zeros(slots.shape, dtype=bool)
    keep[first_positions] = True
    return keep


def group_scans_by_hour(scan_end_times):
    """Group the scans of a raw store by the full hour they are provided at.

    Args:
        scan_end_times: Observation end time of every scan, datetime64, sorted.

    Returns:
        hours: The hours that have at least one scan, datetime64[h], sorted.
        scan_positions: For every hour the positions of its scans in scan_end_times
            (duplicates of a 15-minute slot removed), a list of integer arrays.
    """
    keep = first_scan_per_slot(scan_end_times)
    kept_positions = np.flatnonzero(keep)
    kept_labels = hour_labels(scan_end_times)[keep]

    hours, first_of_hour = np.unique(kept_labels, return_index=True)
    boundaries = list(first_of_hour) + [kept_labels.size]
    scan_positions = []
    for hour_number in range(hours.size):
        start = boundaries[hour_number]
        stop = boundaries[hour_number + 1]
        scan_positions.append(kept_positions[start:stop])
    return hours, scan_positions


def hourly_mean_rate(scan_rates, scan_valid):
    """Average the scans of one hour.

    Args:
        scan_rates: Rain rate of every scan in mm/h, shape (n_scans, n_cells).
        scan_valid: True where a scan has a valid rate for a cell, same shape.

    Returns:
        mean_rate: Mean of the valid scans in mm/h, which is the rain of the hour in mm.
            0 where no scan is valid. float32, shape (n_cells,).
        n_scans: Number of valid scans per cell. int8, shape (n_cells,).
    """
    scan_rates = np.asarray(scan_rates, dtype=np.float64)
    scan_valid = np.asarray(scan_valid, dtype=bool)

    n_scans = scan_valid.sum(axis=0)
    rate_sum = np.where(scan_valid, scan_rates, 0.0).sum(axis=0)

    mean_rate = np.zeros(rate_sum.shape, dtype=np.float32)
    has_scan = n_scans > 0
    mean_rate[has_scan] = rate_sum[has_scan] / n_scans[has_scan]
    return mean_rate, n_scans.astype(np.int8)


def encode_rain_rate(mean_rate, n_scans):
    """Scale the hourly mean rate to the stored integers.

    Args:
        mean_rate: Hourly mean rate in mm/h, shape (n_cells,).
        n_scans: Number of valid scans per cell, shape (n_cells,).

    Returns:
        int16 array, round(mean_rate / RAIN_RATE_SCALE), RAIN_RATE_FILL where n_scans is 0.
    """
    encoded = np.rint(np.asarray(mean_rate) / RAIN_RATE_SCALE).astype(np.int16)
    encoded[np.asarray(n_scans) == 0] = RAIN_RATE_FILL
    return encoded
