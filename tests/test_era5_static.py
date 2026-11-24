"""The helpers that turn the static arrays of the old ERA5 store into a product store."""

import numpy as np
import pytest

from deep_lst.preprocessing.era5_static.store import (
    CLASS_FIELDS,
    CONTINUOUS_FIELDS,
    STORE_DTYPE,
    STORE_FILL_VALUE,
    cells_to_grid,
    check_grid,
    flatten_fields,
)

LAT = np.array([10.0, 9.75, 9.5])
LON = np.array([20.0, 20.25, 20.5, 20.75])
N_CELLS = LAT.size * LON.size


def grid_centers():
    lon_grid, lat_grid = np.meshgrid(LON, LAT)
    return np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)


def test_store_dtype_has_classes_as_int8_and_the_rest_as_float32():
    assert STORE_DTYPE.names == CLASS_FIELDS + CONTINUOUS_FIELDS
    assert all(STORE_DTYPE[name] == np.dtype("i1") for name in CLASS_FIELDS)
    assert all(STORE_DTYPE[name] == np.dtype("<f4") for name in CONTINUOUS_FIELDS)
    assert STORE_FILL_VALUE[: len(CLASS_FIELDS)] == (-1, -1, -1, -1)
    assert all(np.isnan(fill) for fill in STORE_FILL_VALUE[len(CLASS_FIELDS) :])


def test_flatten_fields_removes_the_subarray_shape_of_the_old_store():
    old_dtype = np.dtype([("tvl", "i1", (1, 1)), ("tvh", "i1", (1, 1))])
    old_values = np.zeros(N_CELLS, dtype=old_dtype)
    old_values["tvl"] = np.arange(N_CELLS).reshape(N_CELLS, 1, 1)
    old_values["tvh"] = -1
    columns = flatten_fields(old_values)
    assert set(columns) == {"tvl", "tvh"}
    assert columns["tvl"].shape == (N_CELLS,)
    assert columns["tvl"].tolist() == list(range(N_CELLS))
    assert np.all(columns["tvh"] == -1)


def test_cells_to_grid_places_row_r_at_lat_index_times_n_lon_plus_lon_index():
    columns = {name: np.arange(N_CELLS).astype(STORE_DTYPE[name]) for name in STORE_DTYPE.names}
    grid_values = cells_to_grid(columns, LAT.size, LON.size)
    assert grid_values.shape == (3, 4)
    assert grid_values.dtype == STORE_DTYPE
    assert grid_values["tvh"][0].tolist() == [0, 1, 2, 3]
    assert grid_values["z"][2, 1] == 2 * LON.size + 1


def test_check_grid_accepts_the_cells_of_the_target_grid_in_row_order():
    check_grid(np.arange(N_CELLS), grid_centers(), LAT, LON)


def test_check_grid_rejects_another_cell_order():
    reordered = np.arange(N_CELLS)[::-1]
    with pytest.raises(ValueError, match="in order"):
        check_grid(reordered, grid_centers(), LAT, LON)


def test_check_grid_rejects_other_coordinates():
    shifted = grid_centers()
    shifted[:, 0] += 0.25
    with pytest.raises(ValueError, match="cell centres"):
        check_grid(np.arange(N_CELLS), shifted, LAT, LON)
