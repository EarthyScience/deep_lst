"""preprocessing.rain_over_africa.mapping: cell numbering of the target grid, the pairs of a
mapping (pixel centres inside a cell, nearest-pixel fallback, land cells only), the regridding
of a scan and the round trip through a file. All on small synthetic grids."""

import numpy as np
import pytest

from deep_lst.preprocessing.rain_over_africa.mapping import (
    SEVIRI_PRODUCT_GRID,
    RegridMapping,
    TargetGrid,
    build_mapping,
    cells_to_grid,
    grid_fingerprint,
    pixel_spacing_km,
    regrid_scan,
)

# rows at latitude 0.1, 0.05, 0, -0.05, -0.1 and columns at longitude 0, 0.05, ..., 0.4
SMALL_GRID = TargetGrid(lat_north=0.1, lon_west=0.0, resolution=0.05, n_lat=5, n_lon=9)
ALL_CELLS = np.arange(SMALL_GRID.n_cells)


def fine_source():
    """Source pixels at a quarter-cell offset, two per cell in each direction for inner cells."""
    source_latitude = 0.1125 - 0.025 * np.arange(9)
    source_longitude = 0.0125 + 0.025 * np.arange(16)
    longitude, latitude = np.meshgrid(source_longitude, source_latitude)
    return latitude, longitude


def coarse_source():
    """Source pixels 0.1 degree (11 km) apart, twice the cell size, shifted 0.01 degree east."""
    source_latitude = np.array([0.1, 0.0, -0.1])
    source_longitude = np.array([0.01, 0.11, 0.21])
    longitude, latitude = np.meshgrid(source_longitude, source_latitude)
    return latitude, longitude


def cell_number(row, column):
    return row * SMALL_GRID.n_lon + column


def test_cells_of_the_seviri_grid_are_numbered_from_the_north_west():
    latitude = np.array([80.0, -80.0, 0.0, 0.02, 0.03, 80.1, 0.0])
    longitude = np.array([-80.0, 80.0, 0.0, -0.02, -0.03, 0.0, 80.2])
    cells = SEVIRI_PRODUCT_GRID.cell_of(latitude, longitude)
    centre = 1600 * 3201 + 1600
    assert cells.tolist() == [0, 3201 * 3201 - 1, centre, centre, centre - 3201 - 1, -1, -1]

    centre_latitude, centre_longitude = SEVIRI_PRODUCT_GRID.cell_centres(np.array([0, centre]))
    assert np.allclose(centre_latitude, [80.0, 0.0])
    assert np.allclose(centre_longitude, [-80.0, 0.0])


def test_cell_value_is_the_mean_of_the_pixels_inside_the_cell():
    latitude, longitude = fine_source()
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    assert np.array_equal(mapping.target_cells, ALL_CELLS)
    assert not mapping.cell_is_fallback.any()

    rates = np.random.default_rng(0).uniform(0.0, 10.0, size=latitude.shape).astype(np.float32)
    valid = np.ones(latitude.shape, dtype=bool)
    cell_rates, cell_valid = regrid_scan(rates, valid, mapping)

    pixel_cell = SMALL_GRID.cell_of(latitude, longitude)
    for position, cell in enumerate(mapping.target_cells):
        expected = rates[pixel_cell == cell].mean()
        assert np.isclose(cell_rates[position], expected)
    assert cell_valid.all()
    # an inner cell holds 2 x 2 pixels
    inner_cell = cell_number(row=2, column=3)
    assert (pixel_cell == inner_cell).sum() == 4


def test_only_land_cells_receive_a_value():
    latitude, longitude = fine_source()
    land_cells = np.array([cell_number(1, 1), cell_number(2, 3), cell_number(4, 8)])
    mapping = build_mapping(latitude, longitude, land_cells, SMALL_GRID)
    assert np.array_equal(mapping.target_cells, land_cells)


def test_invalid_pixels_are_left_out_of_the_cell_mean():
    latitude, longitude = fine_source()
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    pixel_cell = SMALL_GRID.cell_of(latitude, longitude)

    rates = np.full(latitude.shape, 2.0, dtype=np.float32)
    valid = np.ones(latitude.shape, dtype=bool)
    partly_valid_cell = cell_number(2, 3)
    empty_cell = cell_number(3, 5)
    rows, columns = np.nonzero(pixel_cell == partly_valid_cell)
    rates[rows[0], columns[0]] = 8.0  # one of the four pixels differs ...
    valid[rows[1], columns[1]] = False  # ... and another one is invalid
    valid[pixel_cell == empty_cell] = False

    cell_rates, cell_valid = regrid_scan(rates, valid, mapping)
    assert np.isclose(cell_rates[partly_valid_cell], (8.0 + 2.0 + 2.0) / 3)
    assert cell_valid[partly_valid_cell]
    assert cell_rates[empty_cell] == 0.0
    assert not cell_valid[empty_cell]


def test_cell_without_pixel_centre_takes_the_nearest_pixel_within_reach():
    latitude, longitude = coarse_source()
    assert np.allclose(pixel_spacing_km(latitude, longitude), 11.12, atol=0.01)
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)

    # the row at latitude 0: pixel centres fall into the columns 0, 2 and 4
    middle_row = [cell_number(2, column) for column in range(9)]
    is_mapped = np.isin(middle_row, mapping.target_cells)
    # columns 6 to 8 are 10 km or more from the last pixel: more than 0.71 x 11 km
    assert is_mapped.tolist() == [True, True, True, True, True, True, False, False, False]
    positions = np.searchsorted(mapping.target_cells, middle_row[:6])
    assert mapping.cell_is_fallback[positions].tolist() == [False, True, False, True, False, True]

    rates = np.arange(9, dtype=np.float32).reshape(3, 3)  # the row at latitude 0 holds 3, 4, 5
    valid = np.ones((3, 3), dtype=bool)
    cell_rates, _ = regrid_scan(rates, valid, mapping)
    assert cell_rates[positions].tolist() == [3.0, 3.0, 4.0, 4.0, 5.0, 5.0]


def test_pixels_without_coordinates_are_ignored():
    latitude, longitude = fine_source()
    latitude[:2, :] = np.nan  # the pixels of the first target row
    longitude[:2, :] = np.nan
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    first_row = [cell_number(0, column) for column in range(9)]
    direct_cells = mapping.target_cells[~mapping.cell_is_fallback]
    assert not np.isin(first_row, direct_cells).any()
    assert np.isfinite(latitude.ravel()[mapping.pair_pixel]).all()


def test_values_are_placed_on_the_full_grid_with_fill_elsewhere():
    latitude, longitude = fine_source()
    land_cells = np.array([cell_number(1, 1), cell_number(2, 3)])
    mapping = build_mapping(latitude, longitude, land_cells, SMALL_GRID)
    grid = cells_to_grid(np.array([7, 9], dtype=np.int16), mapping, fill_value=-32768)
    assert grid.shape == (5, 9)
    assert grid.dtype == np.int16
    assert grid[1, 1] == 7
    assert grid[2, 3] == 9
    assert (grid == -32768).sum() == 43


def test_scan_of_another_grid_is_rejected():
    latitude, longitude = fine_source()
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    wrong_shape = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        regrid_scan(wrong_shape, np.ones((3, 3), dtype=bool), mapping)


def test_fingerprint_tells_shifted_grids_apart():
    latitude, longitude = fine_source()
    same = grid_fingerprint(latitude.astype(np.float32), longitude.astype(np.float32))
    shifted = grid_fingerprint(latitude + 0.0135, longitude)
    assert grid_fingerprint(latitude, longitude) == same
    assert grid_fingerprint(latitude, longitude) != shifted


def test_mapping_survives_the_round_trip_through_a_file(tmp_path):
    latitude, longitude = coarse_source()
    mapping = build_mapping(latitude, longitude, ALL_CELLS, SMALL_GRID)
    path = tmp_path / "mapping.npz"
    mapping.save(path)
    loaded = RegridMapping.load(path)

    assert loaded.source_shape == mapping.source_shape == (3, 3)
    assert loaded.target_shape == mapping.target_shape == (5, 9)
    assert loaded.source_fingerprint == mapping.source_fingerprint
    for name in ("target_cells", "cell_is_fallback", "pair_cell", "pair_pixel"):
        assert np.array_equal(getattr(loaded, name), getattr(mapping, name))
