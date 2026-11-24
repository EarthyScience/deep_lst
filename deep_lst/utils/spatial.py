"""Grids of the stores and of the forecast files.

The anchors of a selection are mapped to store cells in deep_lst/stores/anchors.py; the preload
(deep_lst/dataset/preload.py) does that for training and evaluation. The helpers here serve the
scripts that work with a store grid directly, and the forecast writers.
"""

import numpy as np

from deep_lst.stores import get_data_reader
from deep_lst.utils.io import ExperimentPaths, default_sources


def store_grid(modality, paths=None):
    """LatLonGrid of a modality: the grid of its first store in conf/sources/product.yaml.

    A row of this grid is the cell index used for the modality (row == cell).
    """
    paths = paths or ExperimentPaths()
    store_name = next(iter(default_sources()[modality]["stores"]))
    reader = get_data_reader(store_name, paths.product_store_path(store_name))
    return reader.spatial_grid


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
