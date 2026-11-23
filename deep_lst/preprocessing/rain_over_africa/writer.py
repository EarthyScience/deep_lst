"""Values of one shard of the Rain over Africa (RoA) store, computed from the raw mirror.

A shard of the store holds a fixed number of consecutive hours (96) on the full target grid.
The shards are aligned to the hour count since 1970, not to calendar years, so a shard can
need scans of two yearly raw stores. The scans of all years are therefore listed in one
catalogue, and each hour is computed from the yearly store that holds its scans, with the
regrid mapping of that year's source grid.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from deep_lst.preprocessing.rain_over_africa.aggregation import (
    N_SCANS_FILL,
    RAIN_RATE_FILL,
    STORE_DTYPE,
    clean_rates,
    encode_rain_rate,
    group_scans_by_hour,
    hourly_mean_rate,
)
from deep_lst.preprocessing.rain_over_africa.mapping import RegridMapping, grid_fingerprint, regrid_scan


@dataclass(frozen=True)
class RawYear:
    """One yearly raw store.

    Attributes:
        year: Calendar year of the store.
        scan_end_times: Observation end time of every scan, datetime64, sorted. (n_scans,)
        rain_rate: Array-like, rain_rate[position] is one scan in mm/h on the source grid.
        mapping: The RegridMapping of the source grid of this year.
    """

    year: int
    scan_end_times: np.ndarray
    rain_rate: object
    mapping: RegridMapping


@dataclass(frozen=True)
class ScanCatalogue:
    """The scans of all raw years, grouped by the full hour they are provided at.

    Attributes:
        hours: The hours with at least one scan, datetime64[h], sorted. (n_hours,)
        scan_numbers: For every hour the numbers of its scans in the two arrays below.
        scan_year_number: Position in the list of raw years of every scan. (n_scans,)
        scan_position: Position of every scan in its yearly raw store. (n_scans,)
    """

    hours: np.ndarray
    scan_numbers: list
    scan_year_number: np.ndarray
    scan_position: np.ndarray


def mapping_path(mapping_dir, fingerprint):
    return Path(mapping_dir) / f"roa_to_seviri_land_{fingerprint}.npz"


def open_raw_years(raw_dir, mapping_dir, years):
    """Open the yearly raw stores of the mirror together with the mapping of their source grid."""
    raw_years = []
    mappings = {}
    for year in years:
        store_path = Path(raw_dir) / f"roa_{year}.zarr"
        latitude = zarr.open_array(store_path / "latitude", mode="r")[:]
        longitude = zarr.open_array(store_path / "longitude", mode="r")[:]
        fingerprint = grid_fingerprint(latitude, longitude)
        if fingerprint not in mappings:
            mappings[fingerprint] = RegridMapping.load(mapping_path(mapping_dir, fingerprint))

        # the raw time axis counts milliseconds since 1970
        end_times = zarr.open_array(store_path / "time", mode="r")[:].astype("datetime64[ms]")
        raw_year = RawYear(
            year=year,
            scan_end_times=end_times,
            rain_rate=zarr.open_array(store_path / "y_hat_mu", mode="r"),
            mapping=mappings[fingerprint],
        )
        raw_years.append(raw_year)
    return raw_years


def build_catalogue(raw_years):
    """List the scans of all raw years by the hour they are provided at.

    Args:
        raw_years: The yearly raw stores in ascending order of the year.

    Returns:
        The ScanCatalogue of these years.
    """
    end_times = []
    year_numbers = []
    positions = []
    for year_number, raw_year in enumerate(raw_years):
        n_scans = raw_year.scan_end_times.size
        end_times.append(raw_year.scan_end_times.astype("datetime64[ms]"))
        year_numbers.append(np.full(n_scans, year_number, dtype=np.int64))
        positions.append(np.arange(n_scans, dtype=np.int64))
    end_times = np.concatenate(end_times)
    if np.any(np.diff(end_times.astype(np.int64)) < 0):
        raise ValueError("the scan end times are not sorted across the raw years")

    hours, scan_numbers = group_scans_by_hour(end_times)
    return ScanCatalogue(
        hours=hours,
        scan_numbers=scan_numbers,
        scan_year_number=np.concatenate(year_numbers),
        scan_position=np.concatenate(positions),
    )


def time_index_of_hours(hours):
    """Position of full hours on the hourly time axis of the store (hours since 1970)."""
    return np.asarray(hours).astype("datetime64[h]").astype(np.int64)


def hour_values(raw_years, catalogue, hour_number):
    """Compute the store values of one hour of the catalogue.

    Returns:
        rain_rate: Encoded hourly rain per target cell of the mapping, int16. (n_cells,)
        n_scans: Number of valid scans per target cell, int8. (n_cells,)
        mapping: The RegridMapping the cells refer to.

    Raises:
        ValueError: If the scans of the hour come from two raw years.
    """
    scan_numbers = catalogue.scan_numbers[hour_number]
    year_numbers = np.unique(catalogue.scan_year_number[scan_numbers])
    if year_numbers.size != 1:
        raise ValueError(f"the scans of hour {catalogue.hours[hour_number]} come from two raw years")
    raw_year = raw_years[int(year_numbers[0])]

    scan_rates = []
    scan_valid = []
    for position in catalogue.scan_position[scan_numbers]:
        raw_scan = raw_year.rain_rate[int(position)]
        clean, valid = clean_rates(raw_scan)
        cell_rates, cell_valid = regrid_scan(clean, valid, raw_year.mapping)
        scan_rates.append(cell_rates)
        scan_valid.append(cell_valid)

    mean_rate, n_scans = hourly_mean_rate(np.stack(scan_rates), np.stack(scan_valid))
    rain_rate = encode_rain_rate(mean_rate, n_scans)
    return rain_rate, n_scans, raw_year.mapping


def shard_values(raw_years, catalogue, first_time_index, n_hours):
    """Compute all values of one shard.

    Args:
        raw_years: The yearly raw stores, as given to build_catalogue.
        catalogue: The ScanCatalogue of these years.
        first_time_index: Position of the first hour of the shard on the time axis of the store.
        n_hours: Number of hours in a shard.

    Returns:
        values: Structured array (STORE_DTYPE) of shape (n_hours, n_lat, n_lon), fill values
            where there is no data.
        written_time_indices: Time indices of the hours that have at least one scan.
    """
    n_lat, n_lon = raw_years[0].mapping.target_shape
    values = np.empty((n_hours, n_lat, n_lon), dtype=STORE_DTYPE)
    values["rain_rate"] = RAIN_RATE_FILL
    values["n_scans"] = N_SCANS_FILL

    time_indices = time_index_of_hours(catalogue.hours)
    in_shard = (time_indices >= first_time_index) & (time_indices < first_time_index + n_hours)
    hour_numbers = np.flatnonzero(in_shard)

    for hour_number in hour_numbers:
        rain_rate, n_scans, mapping = hour_values(raw_years, catalogue, int(hour_number))
        hour_offset = int(time_indices[hour_number] - first_time_index)
        cell_rows = mapping.target_cells // n_lon
        cell_columns = mapping.target_cells % n_lon
        values["rain_rate"][hour_offset, cell_rows, cell_columns] = rain_rate
        values["n_scans"][hour_offset, cell_rows, cell_columns] = n_scans

    return values, time_indices[hour_numbers]
