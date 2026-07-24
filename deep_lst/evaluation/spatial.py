from pathlib import Path

import numpy as np
import zarr


def lookup_pixel_coords(zarr_store, unique_spatial_idx: np.ndarray):
    """Look up per-pixel (lat, lon) for the given spatial indices via the store's embedded coords.

    Reads `spatial_idx`, `lat`, `lon` directly from a sampled zarr store. Works for any index
    space (MTG global, equiangular flat at any resolution) because no grid math is performed —
    extract_sampled writes per-pixel coords at extraction time.

    Args:
        zarr_store: An open zarr.Group, or a path/str to one.
        unique_spatial_idx: Sorted-unique 1-D array of spatial indices. Must be a subset of the
            store's `spatial_idx` array (raises ValueError otherwise).

    Returns:
        lats: Unique latitudes covering the bounding box, north-to-south. (H,)
        lons: Unique longitudes covering the bounding box, west-to-east. (W,)
        local_rows: Row indices into `lats` for each spatial idx. (N,)
        local_cols: Col indices into `lons` for each spatial idx. (N,)
        grid_shape: (H, W).
    """
    if isinstance(zarr_store, (str | Path)):
        zarr_store = zarr.open_group(str(zarr_store), mode="r")

    store_spatial_idx = zarr_store["spatial_idx"][:]
    store_lats = zarr_store["lat"][:]
    store_lons = zarr_store["lon"][:]

    positions = np.searchsorted(store_spatial_idx, unique_spatial_idx)
    if positions.max() >= len(store_spatial_idx) or not np.array_equal(
        store_spatial_idx[positions], unique_spatial_idx
    ):
        raise ValueError("Some unique_spatial_idx values are not present in the store's spatial_idx.")

    pixel_lats = store_lats[positions]
    pixel_lons = store_lons[positions]

    return pixel_grid(pixel_lons, pixel_lats)


def pixel_grid(pixel_lons: np.ndarray, pixel_lats: np.ndarray):
    """Bounding-box grid geometry for per-pixel coordinates on a regular (equiangular) grid.

    Same return contract as lookup_pixel_coords, but the per-pixel lat/lon are supplied directly
    (e.g. from the preloaded cell_lonlat) rather than read from a store — no sampled zarr needed.

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
