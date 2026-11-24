"""Raw store values for the analysis scripts.

The same anchor-to-cell mechanism the training and evaluation preloads use
(deep_lst/dataset/preload.py), packaged for one-off analyses: the anchors of a selection are
mapped to the nearest cells of a modality's stores and read as physical, not normalised, values.
"""

import copy

import numpy as np

from deep_lst.dataset.dataset import PATHS, _load_anchor_ids
from deep_lst.dataset.preload import preload_modality
from deep_lst.utils.io import default_sources
from deep_lst.utils.spatial import store_grid


def load_store_columns(modality, spatial, load, anchor="mtg", day_range=None, num_workers=8):
    """Raw store columns of one modality for one spatial selection.

    Args:
        modality: Modality of conf/sources/product.yaml.
        spatial: Selection; its anchors come from indices_<spatial>.npz.
        load: Variables to read by leaf, e.g. {"data": ["LST"]}; they must be listed under the
            stores of the modality.
        anchor: "mtg", or the modality whose grid rows the anchor ids are.
        day_range: (start_day, end_day) in days since 1970, end exclusive. Default: 2016 to 2025.
        num_workers: Threads that read time blocks in parallel.

    Returns:
        columns: {leaf: tensor}; leaves on the hourly axis (n_unique, n_days * 24, C), static
            leaves (n_unique, C), NaN where missing.
        unique_rows: Row in the modality's grid of every unique cell; lon/lat of the cells are
            store_grid(modality).centers[unique_rows].
        inverse: Position in the columns for every anchor, so columns[leaf][inverse] expands back
            to per-anchor order.
        anchor_ids: The anchor ids of the selection.
    """
    anchor_ids = _load_anchor_ids(spatial)
    source = copy.deepcopy(default_sources()[modality])
    source["load"] = load
    anchor_grid = None if anchor == "mtg" else store_grid(anchor)
    columns, inverse, cell_lonlat = preload_modality(
        source, anchor, anchor_ids, PATHS, anchor_grid=anchor_grid, day_range=day_range, num_workers=num_workers
    )
    unique_rows = store_grid(modality).nearest_rows(cell_lonlat[:, 0], cell_lonlat[:, 1])
    return columns, np.asarray(unique_rows), inverse, anchor_ids
