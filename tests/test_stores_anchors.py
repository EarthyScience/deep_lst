"""Anchors of a selection -> lon, lat -> rows of a store, without eodata.

The fixture tests compare with what eodata returned for the ten anchors of
tests/fixtures/readers_v1 (scripts/assets/export_reader_fixtures.py): the centres of their
MajorTom cells and their rows in the store of every modality.
"""

from pathlib import Path

import numpy as np
import pytest

from deep_lst.stores import anchor_lonlat, majortom_centers, map_to_store_rows
from deep_lst.stores.majortom import cells_per_row, row_edge_longitudes, row_latitudes

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "readers_v1"
MODALITIES = ["seviri", "era5", "era5_land", "glofas_v4"]  # the grids of conftest.store_grids
N_CELLS_5KM = 20374696  # cells of the 5 km grid between 85 S and 85 N


def test_centers_of_the_fixture_anchors_are_those_of_eodata():
    fixture = np.load(FIXTURE_DIR / "seviri.npz")
    centers = majortom_centers(fixture["anchor_ids"])
    assert centers.dtype == np.float64
    assert np.array_equal(centers, fixture["anchor_lonlat"])  # bit for bit


def test_rows_run_from_south_to_north_in_equal_steps():
    latitudes = row_latitudes(grid_distance=5)
    assert latitudes[0] >= -85
    assert latitudes[-1] <= 85
    assert 0.0 in latitudes  # one row starts at the equator
    steps = np.diff(latitudes)
    assert np.allclose(steps, steps[0])
    assert 0.0449 < steps[0] < 0.0450  # 5 km in degrees of latitude


def test_rows_have_fewer_cells_towards_the_poles():
    latitudes = row_latitudes(grid_distance=5)
    n_cells = cells_per_row(latitudes, grid_distance=5)
    assert n_cells.sum() == N_CELLS_5KM
    equator = np.searchsorted(latitudes, 0.0)
    assert n_cells[equator] == 8016
    assert n_cells[0] < n_cells[equator] / 10
    assert n_cells.max() == n_cells[equator]


@pytest.mark.parametrize("n_cells", [8, 9, 8016])
def test_row_longitudes_are_evenly_spaced_from_west_to_east(n_cells):
    longitudes = row_edge_longitudes(n_cells)
    assert longitudes.size == n_cells
    assert longitudes.min() >= -180
    assert longitudes.max() < 180
    assert np.allclose(np.diff(longitudes), 360 / n_cells)


def test_cell_ids_count_west_to_east_then_south_to_north():
    latitudes = row_latitudes(grid_distance=5)
    n_cells = cells_per_row(latitudes, grid_distance=5)
    first_row = majortom_centers(np.arange(n_cells[0]))
    assert np.all(np.diff(first_row[:, 0]) > 0)
    assert np.all(first_row[:, 1] == first_row[0, 1])
    next_row = majortom_centers([n_cells[0]])
    assert next_row[0, 1] > first_row[0, 1]
    assert next_row[0, 0] < first_row[-1, 0]


def test_order_and_repeats_of_the_ids_are_kept():
    ids = np.array([12616679, 4483159, 12616679, 16348980])
    centers = majortom_centers(ids)
    assert np.array_equal(centers[0], centers[2])
    for position, cell_id in enumerate(ids):
        assert np.array_equal(centers[position], majortom_centers([cell_id])[0])


@pytest.mark.parametrize("cell_id", [-1, N_CELLS_5KM])
def test_ids_outside_the_grid_are_rejected(cell_id):
    with pytest.raises(ValueError, match="cell ids"):
        majortom_centers([0, cell_id])
    assert majortom_centers([0, N_CELLS_5KM - 1]).shape == (2, 2)


def test_anchor_lonlat_of_mtg_anchors_are_the_majortom_centers():
    fixture = np.load(FIXTURE_DIR / "seviri.npz")
    lonlat = anchor_lonlat("mtg", fixture["anchor_ids"])
    assert np.array_equal(lonlat, fixture["anchor_lonlat"])


@pytest.mark.parametrize("modality", MODALITIES)
def test_mtg_anchors_map_to_the_store_rows_of_eodata(modality, store_grids):
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    rows = map_to_store_rows("mtg", fixture["anchor_ids"], store_grids[modality])
    assert rows.dtype == np.int64
    assert np.array_equal(rows, fixture["store_rows"])


def test_anchors_on_a_store_grid_are_its_rows(store_grids):
    seviri = store_grids["seviri"]
    rng = np.random.default_rng(0)
    anchor_ids = rng.integers(0, len(seviri), size=5000)
    lonlat = anchor_lonlat("seviri", anchor_ids, anchor_grid=seviri)
    lat_index, lon_index = seviri.rows_to_lat_lon_index(anchor_ids)
    assert np.array_equal(lonlat[:, 0], seviri.lon[lon_index])
    assert np.array_equal(lonlat[:, 1], seviri.lat[lat_index])
    rows = map_to_store_rows("seviri", anchor_ids, seviri, anchor_grid=seviri)
    assert np.array_equal(rows, anchor_ids)


def test_seviri_anchors_map_to_the_era5_cell_that_contains_them(store_grids):
    seviri, era5 = store_grids["seviri"], store_grids["era5"]
    # SEVIRI cell at 13.5 N, 2.1 E; the ERA5 cell centred at 13.5 N, 2.0 E contains it
    anchor_id = seviri.nearest_rows(2.1, 13.5)
    rows = map_to_store_rows("seviri", [anchor_id], era5, anchor_grid=seviri)
    lat_index, lon_index = era5.rows_to_lat_lon_index(rows)
    assert np.isclose(era5.lat[lat_index[0]], 13.5)
    assert np.isclose(era5.lon[lon_index[0]], 2.0)
