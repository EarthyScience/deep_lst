"""Spatial anchor-to-store-row mapping for the full (unextracted) eodata stores.

The training/eval selection lives on an *anchor* grid — the global ``MajorTomGrid`` for
training, or the equiangular seviri grid for eval holdout regions. Each modality's full
store is indexed on its own equiangular grid (store row == grid cell index). These helpers
map every anchor cell to the nearest native store row via eodata's grid mapping, so both the
training dataset and the evaluator can gather each modality's nearest cell without any
extracted/sampled store.
"""

import numpy as np
from eodata.grids import build_era5_grid, build_era5_land_grid, build_seviri_grid
from eodata.grids.majortom import MajorTomGrid
from eodata.grids.mapping import SpatialGridMapping
from eodata.grids.spatial import IndexedSubsetGrid

MTG_RESOLUTION = 5

_STORE_GRID_BUILDERS = {
    "seviri": build_seviri_grid,
    "era5": build_era5_grid,
    "era5_land": build_era5_land_grid,
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
    """Nearest-neighbour each anchor cell (anchor_ids in src_grid) to a native dst_grid row.

    Returns an int64 array of length len(anchor_ids); entry i is the store row nearest anchor
    cell anchor_ids[i]. When the anchor grid IS the store grid (eval seviri holdout), the anchor
    ids already are store rows and pass through unchanged. precise=False = centre KDTree, matching
    the site-selection pipeline. Coarse stores repeat rows (many anchors -> one cell).
    """
    anchor_ids = np.asarray(anchor_ids, dtype=np.int64)
    if src_grid.name == dst_grid.name:
        return anchor_ids
    src = IndexedSubsetGrid(src_grid, anchor_ids, name="anchor")
    mapping = SpatialGridMapping.from_nearest_neighbors(src, dst_grid, precise=False, cached=False)
    return np.asarray(mapping.indices, dtype=np.int64)
