"""Shared full-store access for the exploration scripts.

The same anchor-to-store-row mechanism the training and eval preloads use
(dataset.preload_shared), packaged for one-off analyses: anchor ids from an
indices npz are nearest-neighbour mapped to a modality's native store rows and
read raw (unnormalized) into RAM — no extracted/sampled stores.
"""

import inspect

import eodata
import numpy as np
from eodata.registry import get_data_reader_class

from deep_lst.dataset.dataset import (
    DATA_TYPES,
    DATA_TYPES_ENABLE,
    PATHS,
    _load_anchor_ids,
    _preload_columns,
)
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

# Full-store names, mirroring the data version pinned in conf/train.yaml.
SEVIRI_STORE = "seviri_2016_2024"
ERA5_STORE = "era5_2014_2025"
ERA5_LAND_STORE = "era5_land_2016_2025"


def open_store_reader(modality, store, data_types):
    """Full-store reader with exactly the `data_types` enabled (data is always read).

    Enable-flags for data types this modality's reader doesn't accept (ERA5 has no
    metadata/data_daily) are filtered out by the reader's __init__ signature.
    """
    reader_kwargs = {"average_hourly": False}
    for data_type, flag in DATA_TYPES_ENABLE.items():
        reader_kwargs[flag] = data_type in data_types
    accepted = inspect.signature(get_data_reader_class(modality)).parameters
    reader_kwargs = {key: value for key, value in reader_kwargs.items() if key in accepted}
    return eodata.get_data_reader(modality, store=str(PATHS.eozarr_store_path(store)), **reader_kwargs)


def load_store_columns(modality, store, spatial, load, anchor="mtg", day_range=None, num_workers=8):
    """Raw full-store columns for one spatial selection.

    Anchor ids from indices_{spatial}.npz are mapped to the modality's native store
    rows (deduped). load maps each data type to its variable names, e.g. {"data": ["LST"]};
    day_range restricts the read to (start_day, end_day) store days (default: the
    full preload axis).

    Returns (columns, unique_rows, inverse, anchor_ids): hourly/daily data types ->
    (n_unique, n_days * 24, C), static data types -> (n_unique, C), NaN where missing.
    inverse maps each anchor position to its row in the columns, so
    columns[data_type][inverse] expands back to per-anchor order. Cell lon/lat:
    store_grid(modality).centers[unique_rows].
    """
    anchor_ids = _load_anchor_ids(spatial)
    rows = map_to_store_rows(anchor_grid(anchor), anchor_ids, store_grid(modality))
    unique_rows, inverse = np.unique(rows, return_inverse=True)
    reader = open_store_reader(modality, store, load)
    select = {
        data_type: [list(getattr(reader, DATA_TYPES[data_type]).dtype.fields).index(var) for var in names]
        for data_type, names in load.items()
    }
    columns = _preload_columns(reader, None, unique_rows, select, num_workers=num_workers, day_range=day_range)
    return columns, unique_rows, inverse, anchor_ids
