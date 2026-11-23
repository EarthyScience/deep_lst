"""LatLonGrid: store rows, grid positions and coordinates, without eodata.

The fixture tests compare with what eodata returned for the ten anchors of
tests/fixtures/readers_v1 (scripts/assets/export_reader_fixtures.py).
"""

from pathlib import Path

import numpy as np
import pytest

from deep_lst.stores import LatLonGrid

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "readers_v1"

# modality -> first latitude, first longitude, resolution, n_lat, n_lon of its store grid
STORE_GRIDS = {
    "seviri": (80.0, -80.0, 0.05, 3201, 3201),
    "era5": (90.0, -180.0, 0.25, 721, 1440),
    "era5_land": (90.0, -180.0, 0.1, 1801, 3600),
    "glofas_v4": (89.975, -179.975, 0.05, 3000, 7200),
}


def small_grid():
    lat = np.array([10.0, 9.0, 8.0])
    lon = np.array([20.0, 21.0, 22.0, 23.0])
    return LatLonGrid(lat, lon, name="small")


def store_grid(modality):
    lat_first, lon_first, resolution, n_lat, n_lon = STORE_GRIDS[modality]
    lat = lat_first - resolution * np.arange(n_lat)
    lon = lon_first + resolution * np.arange(n_lon)
    return LatLonGrid(lat, lon, name=modality)


def test_shape_length_and_steps():
    grid = small_grid()
    assert grid.shape == (3, 4)
    assert len(grid) == 12
    assert grid.lat_step == -1.0
    assert grid.lon_step == 1.0


def test_row_is_lat_index_times_n_lon_plus_lon_index():
    grid = small_grid()
    lat_index, lon_index = grid.rows_to_lat_lon_index([0, 3, 4, 11])
    assert lat_index.tolist() == [0, 0, 1, 2]
    assert lon_index.tolist() == [0, 3, 0, 3]
    rows = grid.lat_lon_index_to_rows(lat_index, lon_index)
    assert rows.tolist() == [0, 3, 4, 11]


def test_centers_are_lon_lat_in_row_order():
    grid = small_grid()
    assert grid.centers.shape == (12, 2)
    assert grid.centers[0].tolist() == [20.0, 10.0]
    assert grid.centers[3].tolist() == [23.0, 10.0]
    assert grid.centers[4].tolist() == [20.0, 9.0]
    assert grid.centers[11].tolist() == [23.0, 8.0]


def test_nearest_rows_of_the_cell_centers_are_the_rows():
    grid = small_grid()
    rows = np.arange(len(grid))
    nearest = grid.nearest_rows(grid.centers[:, 0], grid.centers[:, 1])
    assert np.array_equal(nearest, rows)


def test_nearest_rows_rounds_to_the_closest_center():
    grid = small_grid()
    lon = np.array([20.4, 21.6, 22.9])
    lat = np.array([9.6, 9.4, 8.2])
    assert grid.nearest_rows(lon, lat).tolist() == [0, 6, 11]


def test_points_outside_get_the_edge_cell():
    grid = small_grid()
    lon = np.array([0.0, 100.0])
    lat = np.array([50.0, -50.0])
    assert grid.nearest_rows(lon, lat).tolist() == [0, 11]


def test_uneven_axis_is_rejected():
    lat = np.array([10.0, 9.0, 7.0])
    lon = np.array([20.0, 21.0, 22.0])
    with pytest.raises(ValueError, match="lat axis"):
        LatLonGrid(lat, lon, name="uneven")


@pytest.mark.parametrize("modality", list(STORE_GRIDS))
def test_centers_of_the_fixture_rows_are_the_eodata_cell_coordinates(modality):
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    grid = store_grid(modality)
    centers = grid.centers[fixture["store_rows"]]
    assert np.allclose(centers, fixture["cell_lonlat"], rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("modality", list(STORE_GRIDS))
def test_nearest_rows_of_the_fixture_anchors_are_the_eodata_rows(modality):
    fixture = np.load(FIXTURE_DIR / f"{modality}.npz")
    grid = store_grid(modality)
    anchor_lonlat = fixture["anchor_lonlat"]
    rows = grid.nearest_rows(anchor_lonlat[:, 0], anchor_lonlat[:, 1])
    assert np.array_equal(rows, fixture["store_rows"])
