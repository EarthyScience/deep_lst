"""Cell centres of the MajorTom grid, the grid of the training anchors.

The MajorTom grid divides the globe into rows of equal height in latitude; every row is divided
into as many cells as fit around the globe at its latitude with the grid distance (5 km for the
anchors of deep_lst). A cell id counts the cells row by row from south to north, within a row
from west to east. Only the rows between 85 S and 85 N belong to the grid.

The index files (indices_<selection>.npz) hold cell ids of the grid as eodata defines it
(eodata.grids.majortom.MajorTomGrid), whose centres differ slightly from those of the original
MajorTom code. This module reproduces eodata's centres bit for bit; it was compared with eodata
for every id of the index files and for 200,000 random ids on 2026-11-24. The arithmetic below
therefore follows eodata operation by operation, including the sort of the longitudes of a row,
and must not be simplified without repeating that comparison.
"""

import math

import numpy as np

RADIUS_EQUATOR = 6378.137  # km; the grid treats the Earth as a sphere of this radius
LATITUDE_RANGE = (-85, 85)  # rows outside are not part of the grid


def row_latitudes(grid_distance):
    """Latitude of the southern edge of every row, south to north. (n_rows,)"""
    pole_to_pole = math.pi * RADIUS_EQUATOR
    n_rows_pole_to_pole = int(math.ceil(pole_to_pole / grid_distance))
    latitudes = np.linspace(-90, 90, n_rows_pole_to_pole, endpoint=False)
    latitudes = np.sort(np.mod(latitudes, 180) - 90)  # shifted so that one row starts at 0
    in_range = (latitudes >= LATITUDE_RANGE[0]) & (latitudes <= LATITUDE_RANGE[1])
    return latitudes[in_range]


def cells_per_row(latitudes, grid_distance):
    """Number of cells in the rows with the given southern edges. (n_rows,)"""
    circumference = 2 * math.pi * RADIUS_EQUATOR * np.cos(np.radians(latitudes))
    return np.ceil(circumference / grid_distance).astype(int)


def row_edge_longitudes(n_cells):
    """Longitude of the western edge of every cell of a row with n_cells cells, west to east."""
    cell_width = 360 / n_cells
    longitudes = np.arange(n_cells, dtype=float) * cell_width - 180
    longitudes = np.mod(longitudes, 360) - 180  # eodata moves the date line to the middle
    return np.sort(longitudes)


def majortom_centers(cell_ids, grid_distance=5):
    """Lon, lat of the centres of MajorTom cells.

    Args:
        cell_ids: Cell ids, as stored in src_indices of an index file.
        grid_distance: Grid distance in km.

    Returns:
        float64 array (n, 2) with lon, lat in degrees, in the order of cell_ids.

    Raises:
        ValueError: A cell id is not on the grid.
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    latitudes = row_latitudes(grid_distance)
    n_cells = cells_per_row(latitudes, grid_distance)
    row_starts = np.concatenate([[0], np.cumsum(n_cells)])  # id of the first cell of every row
    n_cells_total = row_starts[-1]
    if cell_ids.min() < 0 or cell_ids.max() >= n_cells_total:
        raise ValueError(f"MajorTom cell ids must be in 0 .. {n_cells_total - 1}")

    rows = np.searchsorted(row_starts, cell_ids, side="right") - 1
    positions = cell_ids - row_starts[rows]  # position of the cell within its row
    row_height = latitudes[1] - latitudes[0]

    centers = np.empty((cell_ids.size, 2), dtype=np.float64)
    for row in np.unique(rows):
        in_row = rows == row
        edge_longitudes = row_edge_longitudes(n_cells[row])
        half_cell_width = 360 / 2 / n_cells[row]
        centers[in_row, 0] = edge_longitudes[positions[in_row]] + half_cell_width
        centers[in_row, 1] = latitudes[row] + row_height / 2
    return centers
