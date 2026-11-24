"""From the anchors of a selection to the rows of a store.

A selection (indices_<selection>.npz) lists anchor ids. They are either cells of the MajorTom
grid (anchor "mtg": the training sites and the site holdouts) or rows of the grid of a modality
(anchor "seviri": the dense evaluation regions). Every store is read at the cell whose centre
is nearest to the anchor.
"""

import numpy as np

from deep_lst.stores.majortom import majortom_centers

MTG_GRID_DISTANCE = 5  # km, grid distance of the MajorTom anchors


def anchor_lonlat(anchor, anchor_ids, anchor_grid=None):
    """Lon, lat of the anchors of a selection.

    Args:
        anchor: "mtg", or the name of the modality whose grid rows the ids are.
        anchor_ids: Anchor ids of the selection.
        anchor_grid: LatLonGrid of that modality; not needed for "mtg".

    Returns:
        float64 array (n, 2) with lon, lat in degrees.
    """
    if anchor == "mtg":
        return majortom_centers(anchor_ids, MTG_GRID_DISTANCE)
    lat_index, lon_index = anchor_grid.rows_to_lat_lon_index(anchor_ids)
    return np.stack([anchor_grid.lon[lon_index], anchor_grid.lat[lat_index]], axis=1)


def map_to_store_rows(anchor, anchor_ids, store_grid, anchor_grid=None):
    """Row of the store cell nearest to every anchor.

    Args:
        anchor: "mtg", or the name of the modality whose grid rows the ids are.
        anchor_ids: Anchor ids of the selection.
        store_grid: LatLonGrid of the store to read (reader.spatial_grid).
        anchor_grid: LatLonGrid of the anchor modality; not needed for "mtg".

    Returns:
        int64 array (n,) of rows in store_grid, in the order of anchor_ids. Several anchors can
        share a row of a coarse store.
    """
    lonlat = anchor_lonlat(anchor, anchor_ids, anchor_grid)
    return store_grid.nearest_rows(lonlat[:, 0], lonlat[:, 1])
