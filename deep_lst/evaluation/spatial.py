import numpy as np


def pixel_grid(pixel_lons: np.ndarray, pixel_lats: np.ndarray):
    """Bounding-box grid geometry for per-pixel coordinates on a regular (equiangular) grid.

    The per-pixel lat/lon are supplied directly (e.g. from the preloaded cell_lonlat)
    rather than read from a store — no sampled zarr needed.

    Returns:
        lats: Unique latitudes covering the bounding box, north-to-south. (H,)
        lons: Unique longitudes covering the bounding box, west-to-east. (W,)
        local_rows: Row indices into `lats` for each pixel. (N,)
        local_cols: Col indices into `lons` for each pixel. (N,)
        grid_shape: (H, W).
    """
    unique_lats_asc = np.unique(pixel_lats)
    lats = unique_lats_asc[::-1]  # north-to-south convention (matches old EquiangularGrid orientation)
    lons = np.unique(pixel_lons)

    local_rows = (len(lats) - 1) - np.searchsorted(unique_lats_asc, pixel_lats)
    local_cols = np.searchsorted(lons, pixel_lons)
    grid_shape = (len(lats), len(lons))

    return lats, lons, local_rows, local_cols, grid_shape
