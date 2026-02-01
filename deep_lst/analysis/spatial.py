import numpy as np
from wedata.grids import EquiangularGrid


def build_local_grid(spatial_idx: np.ndarray):
    """Convert flat spatial indices to row/col and extract the local lat/lon grid.

    Args:
        spatial_idx: 1D array of flat indices into the global SEVIRI grid. (N_spatial,)
        col_extent:  Number of columns in the global grid (from config).

    Returns:
        local_lats: 1D lat coordinates of the local bounding box. (n_local_rows,)
        local_lons: 1D lon coordinates of the local bounding box. (n_local_cols,)
        local_rows: Row indices relative to the local bbox. (N_spatial,)
        local_cols: Col indices relative to the local bbox. (N_spatial,)
        grid_shape: (n_local_rows, n_local_cols)
    """

    # Build global grid and reshape centers to 2D
    grid = EquiangularGrid(
        resolution=0.05,
        latitude_bounds=(80.0, -80.0),
        longitude_bounds=(-80.0, 80.0),
        include_ymin=True,
        include_xmax=True,
    )
    n_rows, n_cols = grid.get_shape()

    # centers: (n_rows*n_cols, 2) with columns [lon, lat]
    all_lons = grid.centers[:, 0].reshape(n_rows, n_cols)
    all_lats = grid.centers[:, 1].reshape(n_rows, n_cols)

    rows = spatial_idx // n_cols
    cols = spatial_idx % n_cols
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    # Slice to local bbox — regular grid so lat/lon are separable
    local_lats = all_lats[row_min : row_max + 1, 0]
    local_lons = all_lons[0, col_min : col_max + 1]

    local_rows = rows - row_min
    local_cols = cols - col_min
    grid_shape = (row_max - row_min + 1, col_max - col_min + 1)

    return local_lats, local_lons, local_rows, local_cols, grid_shape
