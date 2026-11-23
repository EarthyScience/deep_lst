"""Check one hour of the Rain over Africa (RoA) preprocessing on real data.

For one hour of the raw mirror the script computes the store values with the functions of
deep_lst.preprocessing.rain_over_africa and makes three comparisons:

  1. Random cells against a direct computation that does not use the regrid mapping:
       - a cell with pixel centres: mean of the valid raw pixels within half a cell of the
         cell centre, per scan, then the mean over the scans
       - a cell filled from the nearest pixel: the raw pixel with the smallest distance to
         the cell centre
     The cell centres are read from the lat / lon arrays of the store, so this also covers
     the orientation of the grid.
  2. If the shard of the hour is written: the stored values against the computed values,
     every cell of the grid, both fields.
  3. Maps for the eye, saved in <output_dir>/checks: the rain of the hour, and the cells
     that receive a value (pixel centres inside the cell or nearest pixel).

Run after scripts/preprocessing/preprocess_roa.sh setup, with the same environment:

    cd /Net/Groups/BGI/people/mwessel/deep_lst
    PYTHONPATH=. ../eodata_main/.venv/bin/python scripts/preprocessing/check_roa_hour.py --hour 2020-08-15T15
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from deep_lst.preprocessing.rain_over_africa.aggregation import (
    N_SCANS_FILL,
    RAIN_RATE_FILL,
    RAIN_RATE_SCALE,
)
from deep_lst.preprocessing.rain_over_africa.mapping import cells_to_grid, chord_to_km, unit_vectors
from deep_lst.preprocessing.rain_over_africa.writer import (
    build_catalogue,
    hour_values,
    open_raw_years,
    time_index_of_hours,
)

RAW_DIR = Path("/Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa")
OUTPUT_DIR = Path("/Net/Groups/BGI/tscratch/mwessel/data/rain_over_africa_product")
STORE_NAME = "roa_land_expected_hourly_acc.zarr"
HALF_CELL = 0.025  # degrees
N_SAMPLES = 300  # cells of each kind
SEED = 20261120


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hour", default="2020-08-15T15", help="full hour to check, UTC")
    parser.add_argument("--raw_dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def raw_year_of_hour(hour):
    """Yearly raw store that holds the scans of an hour (they end in the hour before it)."""
    last_minute = hour - np.timedelta64(1, "m")
    return int(str(last_minute.astype("datetime64[Y]")))


def usable_rates(pixels):
    """Valid raw rates, written out here independently of aggregation.clean_rates."""
    return np.isfinite(pixels) & (pixels >= -1e-6) & (pixels <= 200.0)


def direct_value_of_box_cell(scans, latitude, longitude, cell_latitude, cell_longitude):
    """Hourly value of a cell from the raw pixels within half a cell of its centre."""
    in_box = (np.abs(latitude - cell_latitude) < HALF_CELL) & (np.abs(longitude - cell_longitude) < HALF_CELL)
    scan_means = []
    for scan in scans:
        pixels = scan[in_box]
        usable = usable_rates(pixels)
        if usable.any():
            scan_means.append(np.maximum(pixels[usable], 0.0).mean(dtype=np.float64))
    if len(scan_means) == 0:
        return RAIN_RATE_FILL, 0
    return int(np.rint(np.mean(scan_means) / RAIN_RATE_SCALE)), len(scan_means)


def direct_value_of_nearest_pixel(scans, pixel_vectors, cell_latitude, cell_longitude):
    """Hourly value of a cell from the raw pixel closest to its centre, and the distance in km."""
    cell_vector = unit_vectors(cell_latitude, cell_longitude)
    chord = np.linalg.norm(pixel_vectors - cell_vector, axis=-1)
    row, column = np.unravel_index(np.nanargmin(chord), chord.shape)
    pixels = scans[:, row, column]
    usable = usable_rates(pixels)
    distance = float(chord_to_km(chord[row, column]))
    if not usable.any():
        return RAIN_RATE_FILL, 0, distance
    mean_rate = np.maximum(pixels[usable], 0.0).mean(dtype=np.float64)
    return int(np.rint(mean_rate / RAIN_RATE_SCALE)), int(usable.sum()), distance


def compare_with_direct_computation(scans, latitude, longitude, store_latitude, store_longitude, mapping, grids):
    """Comparison 1. Returns True when all sampled cells agree."""
    rain_grid, n_scans_grid = grids
    random = np.random.default_rng(SEED)
    n_lon = mapping.target_shape[1]
    box_cells = random.choice(mapping.target_cells[~mapping.cell_is_fallback], size=N_SAMPLES, replace=False)
    nearest_cells = random.choice(mapping.target_cells[mapping.cell_is_fallback], size=N_SAMPLES, replace=False)

    n_box_equal = 0
    for cell in box_cells:
        row, column = divmod(int(cell), n_lon)
        expected = direct_value_of_box_cell(scans, latitude, longitude, store_latitude[row], store_longitude[column])
        found = (int(rain_grid[row, column]), int(n_scans_grid[row, column]))
        n_box_equal += int(found == expected)

    pixel_vectors = unit_vectors(latitude.astype(np.float64), longitude.astype(np.float64))
    n_nearest_equal = 0
    largest_distance = 0.0
    for cell in nearest_cells:
        row, column = divmod(int(cell), n_lon)
        rate, n_scans, distance = direct_value_of_nearest_pixel(
            scans, pixel_vectors, store_latitude[row], store_longitude[column]
        )
        found = (int(rain_grid[row, column]), int(n_scans_grid[row, column]))
        n_nearest_equal += int(found == (rate, n_scans))
        largest_distance = max(largest_distance, distance)

    print(f"1. cells with pixel centres: {n_box_equal} of {N_SAMPLES} equal the direct computation")
    print(
        f"1. cells from the nearest pixel: {n_nearest_equal} of {N_SAMPLES} equal the direct computation, "
        f"largest distance {largest_distance:.1f} km"
    )
    return n_box_equal == N_SAMPLES and n_nearest_equal == N_SAMPLES


def compare_with_store(store_path, hour, grids):
    """Comparison 2. Returns True when the stored hour equals the computed one, or is not written yet."""
    rain_grid, n_scans_grid = grids
    data = zarr.open_array(store_path / "data", mode="r")
    stored = data[int(time_index_of_hours(hour))]
    if (stored["n_scans"] == N_SCANS_FILL).all():
        print("2. the hour is not written to the store yet, nothing to compare")
        return True
    n_rain_different = int((stored["rain_rate"] != rain_grid).sum())
    n_scans_different = int((stored["n_scans"] != n_scans_grid).sum())
    print(
        f"2. stored against computed, all {rain_grid.size} cells: {n_rain_different} differ in rain_rate, "
        f"{n_scans_different} differ in n_scans"
    )
    return n_rain_different == 0 and n_scans_different == 0


def map_window(grid_with_values):
    """Rows and columns that contain values, as slices."""
    rows = np.flatnonzero(grid_with_values.any(axis=1))
    columns = np.flatnonzero(grid_with_values.any(axis=0))
    return slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1)


def map_extent(store_latitude, store_longitude, row_slice, column_slice):
    return [
        store_longitude[column_slice][0],
        store_longitude[column_slice][-1],
        store_latitude[row_slice][-1],
        store_latitude[row_slice][0],
    ]


def save_coverage_map(mapping, store_latitude, store_longitude, path):
    """Map of the target cells with a value: 1 = pixel centres inside, 2 = nearest pixel."""
    kind = np.ones(mapping.target_cells.size, dtype=np.int8)
    kind[mapping.cell_is_fallback] = 2
    kind_grid = cells_to_grid(kind, mapping, fill_value=np.int8(0))
    row_slice, column_slice = map_window(kind_grid > 0)

    figure, axis = plt.subplots(figsize=(10, 9))
    colours = matplotlib.colors.ListedColormap(["white", "#3b6ea5", "#d1495b"])
    axis.imshow(
        kind_grid[row_slice, column_slice],
        extent=map_extent(store_latitude, store_longitude, row_slice, column_slice),
        cmap=colours,
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.set_title(
        f"RoA on the SEVIRI land grid, source grid {mapping.source_fingerprint}\n"
        f"blue: pixel centres inside the cell ({int((kind == 1).sum())}), "
        f"red: nearest pixel ({int((kind == 2).sum())})"
    )
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_rain_map(rain_grid, hour, store_latitude, store_longitude, path):
    """Map of the rain of one hour in mm; cells without a value stay white."""
    has_value = rain_grid != RAIN_RATE_FILL
    row_slice, column_slice = map_window(has_value)
    rain = np.where(has_value, rain_grid * RAIN_RATE_SCALE, np.nan)[row_slice, column_slice]

    figure, axis = plt.subplots(figsize=(10, 9))
    scale = matplotlib.colors.SymLogNorm(linthresh=0.1, vmin=0.0, vmax=30.0)
    image = axis.imshow(
        rain,
        extent=map_extent(store_latitude, store_longitude, row_slice, column_slice),
        cmap="YlGnBu",
        norm=scale,
        interpolation="nearest",
    )
    axis.set_xlabel("longitude")
    axis.set_ylabel("latitude")
    axis.set_title(f"RoA rain in the hour before {hour} UTC")
    figure.colorbar(image, ax=axis, shrink=0.7, label="mm")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_arguments()
    hour = np.datetime64(args.hour, "h")
    year = raw_year_of_hour(hour)
    store_path = args.output_dir / STORE_NAME
    check_dir = args.output_dir / "checks"
    check_dir.mkdir(parents=True, exist_ok=True)

    raw_years = open_raw_years(args.raw_dir, args.output_dir / "mappings", [year])
    catalogue = build_catalogue(raw_years)
    hour_number = int(np.flatnonzero(catalogue.hours == hour)[0])
    rain_rate, n_scans, mapping = hour_values(raw_years, catalogue, hour_number)
    rain_grid = cells_to_grid(rain_rate, mapping, fill_value=np.int16(RAIN_RATE_FILL))
    n_scans_grid = cells_to_grid(n_scans, mapping, fill_value=np.int8(N_SCANS_FILL))

    raw_year = raw_years[0]
    scan_positions = catalogue.scan_position[catalogue.scan_numbers[hour_number]]
    scans = np.stack([raw_year.rain_rate[int(position)] for position in scan_positions])
    end_times = raw_year.scan_end_times[scan_positions]
    print(f"hour {hour}: {scans.shape[0]} scans ending at {[str(end_time)[11:19] for end_time in end_times]}")
    has_value = rain_grid != RAIN_RATE_FILL
    print(
        f"cells with a value: {int(has_value.sum())} of {mapping.target_cells.size} mapped cells; "
        f"mean {rain_grid[has_value].mean() * RAIN_RATE_SCALE:.4f} mm, max {rain_grid.max() * RAIN_RATE_SCALE:.2f} mm, "
        f"cells with rain >= 0.1 mm: {int((rain_grid >= 10).sum())}"
    )

    raw_store = args.raw_dir / f"roa_{year}.zarr"
    latitude = zarr.open_array(raw_store / "latitude", mode="r")[:]
    longitude = zarr.open_array(raw_store / "longitude", mode="r")[:]
    store_latitude = zarr.open_array(store_path / "lat", mode="r")[:]
    store_longitude = zarr.open_array(store_path / "lon", mode="r")[:]
    grids = (rain_grid, n_scans_grid)

    direct_ok = compare_with_direct_computation(
        scans, latitude, longitude, store_latitude, store_longitude, mapping, grids
    )
    store_ok = compare_with_store(store_path, hour, grids)

    coverage_path = check_dir / f"coverage_{mapping.source_fingerprint}.png"
    rain_path = check_dir / f"rain_{hour}.png"
    save_coverage_map(mapping, store_latitude, store_longitude, coverage_path)
    save_rain_map(rain_grid, hour, store_latitude, store_longitude, rain_path)
    print(f"3. maps: {coverage_path} and {rain_path}")

    if not (direct_ok and store_ok):
        raise SystemExit("CHECK FAILED")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
