"""Spatial anchor-to-store-row mapping for the full (unextracted) eodata stores.

The training/eval selection lives on an *anchor* grid — the global ``MajorTomGrid`` for
training, or the equiangular seviri grid for eval holdout regions. Each modality's full
store is indexed on its own equiangular grid (store row == grid cell index). These helpers
map every anchor cell to the nearest native store row via eodata's grid mapping, so both the
training dataset and the evaluator can gather each modality's nearest cell without any
extracted/sampled store.
"""

import numpy as np
from eodata.grids import build_era5_grid, build_era5_land_grid, build_glofas_v4_grid, build_seviri_grid
from eodata.grids.majortom import MajorTomGrid
from eodata.grids.mapping import SpatialGridMapping
from eodata.grids.spatial import IndexedSubsetGrid

MTG_RESOLUTION = 5

_STORE_GRID_BUILDERS = {
    "seviri": build_seviri_grid,
    "era5": build_era5_grid,
    "era5_land": build_era5_land_grid,
    "glofas_v4": build_glofas_v4_grid,
}


def store_grid(modality):
    """Native equiangular grid a modality's full store is indexed on (cell index == store row)."""
    try:
        return _STORE_GRID_BUILDERS[modality]()
    except KeyError as error:
        raise ValueError(f"No native grid builder for modality {modality!r}") from error


def anchor_grid(name):
    """Grid the anchor cells live on: 'mtg' -> global MajorTomGrid; else a modality's native
    grid (eval holdout regions are given as seviri equiangular pixels)."""
    if name == "mtg":
        return MajorTomGrid(MTG_RESOLUTION)
    return store_grid(name)


def map_to_store_rows(src_grid, anchor_ids, dst_grid):
    """Nearest-neighbour each anchor cell (anchor_ids in src_grid) to a native dst_grid row."""
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    if src_grid.name == dst_grid.name:
        return anchor_ids
    src = IndexedSubsetGrid(src_grid, anchor_ids, name="anchor")
    mapping = SpatialGridMapping.from_nearest_neighbors(src, dst_grid, precise=False, cached=False)
    return np.asarray(mapping.indices, dtype=np.int64)


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
