"""Preload of the product stores: the variables of every modality for the anchors of a selection.

A modality of the config (conf/sources/product.yaml) names its stores and the variables read
from each (`stores`) and sorts the variables into leaves by their role (`load`):

    data                      values on the hourly axis; values of a daily store are repeated
    data_daily                daily values of an hourly modality, repeated over their 24 hours
    data_static               continuous values without time axis
    data_static_categorical   class codes without time axis
    metadata                  quality fields on the hourly axis, kept as stored

The preload reads every store once for the unique store cells of the anchors and returns, per
modality, one array per leaf: (n_cells, n_hours, n_variables) for the leaves on the hourly axis
and (n_cells, n_variables) for the static ones, with NaN where a value is missing. The leaves
data, data_daily and data_static are normalised with the statistics of the modality; class
codes are int8 with MISSING_CLASS where the store has no value.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from deep_lst.stores import T1D, get_data_reader, map_to_store_rows
from deep_lst.utils.normalisation import VariableNormalizer

NORMALISED_LEAVES = ("data", "data_daily", "data_static")
STATIC_LEAVES = ("data_static", "data_static_categorical")
CATEGORICAL_LEAF = "data_static_categorical"
MISSING_CLASS = -1  # class code of a cell without a value

PRELOAD_START_DAY = T1D.date_to_index("2016-01-01")
PRELOAD_END_DAY = T1D.date_to_index("2026-01-01")  # exclusive
SHARDS_PER_BLOCK = 8  # shard files read by one thread in one go


@dataclass
class LeafTarget:
    """Where the variables that one store contributes to one leaf come from and go to."""

    leaf: str
    store_channels: list[int]  # positions of the variables among the variables of the store
    leaf_channels: list[int]  # positions of the same variables in the leaf
    normalizer: VariableNormalizer | None


def open_readers(source, paths):
    """{store name: reader} for the stores of a modality; they must share one grid."""
    readers = {}
    for store_name in source["stores"]:
        readers[store_name] = get_data_reader(store_name, paths.product_store_path(store_name))
    grids = [reader.spatial_grid for reader in readers.values()]
    for grid in grids[1:]:
        same_grid = np.array_equal(grid.lat, grids[0].lat) and np.array_equal(grid.lon, grids[0].lon)
        if not same_grid:
            raise ValueError(f"The stores {list(readers)} of one modality are not on the same grid")
    return readers


def variable_stores(source):
    """{variable: store name} for the variables of `load`; every variable needs exactly one store."""
    stores_of = {}
    for store_name, variables in source["stores"].items():
        for variable in variables:
            stores_of.setdefault(variable, []).append(store_name)
    assignment = {}
    for leaf, variables in source["load"].items():
        for variable in variables:
            candidates = stores_of.get(variable, [])
            if len(candidates) != 1:
                raise ValueError(f"{leaf}.{variable} must be listed under exactly one store, found {candidates}")
            assignment[variable] = candidates[0]
    return assignment


def leaf_targets(source, store_name, reader, stats_zarr):
    """The LeafTargets of one store: one for every leaf that takes variables from it."""
    assignment = variable_stores(source)
    transform = source.get("transform")
    targets = []
    for leaf, variables in source["load"].items():
        from_this_store = [variable for variable in variables if assignment[variable] == store_name]
        if not from_this_store:
            continue
        if reader.is_static != (leaf in STATIC_LEAVES):
            raise ValueError(f"The leaf {leaf!r} cannot take {from_this_store} from the store {store_name!r}")
        normalizer = None
        if stats_zarr is not None and leaf in NORMALISED_LEAVES:
            leaf_transform = transform if leaf == "data" else None
            normalizer = VariableNormalizer(stats_zarr, from_this_store, leaf_transform)
        target = LeafTarget(
            leaf=leaf,
            store_channels=[reader.variables.index(variable) for variable in from_this_store],
            leaf_channels=[variables.index(variable) for variable in from_this_store],
            normalizer=normalizer,
        )
        targets.append(target)
    return targets


def allocate_leaves(load, n_cells, n_hours):
    """One empty array per leaf of `load`."""
    leaves = {}
    for leaf, variables in load.items():
        if leaf == CATEGORICAL_LEAF:
            leaves[leaf] = np.empty((n_cells, len(variables)), dtype=np.int8)
        elif leaf in STATIC_LEAVES:
            leaves[leaf] = np.empty((n_cells, len(variables)), dtype=np.float32)
        else:
            leaves[leaf] = np.empty((n_cells, n_hours, len(variables)), dtype=np.float32)
    return leaves


def target_values(series, target):
    """The variables of a target out of what the store returned, in the form the leaf holds."""
    values = series[..., target.store_channels]
    if target.normalizer is not None:
        values = target.normalizer.normalise(values)
    if target.leaf == CATEGORICAL_LEAF:
        values = np.where(np.isnan(values), MISSING_CLASS, values).astype(np.int8)
    return values


def time_blocks(reader, time_start, time_stop):
    """Split the time steps time_start .. time_stop - 1 into blocks of SHARDS_PER_BLOCK shards."""
    shard_blocks = reader.shard_blocks(time_start, time_stop)
    blocks = []
    for first in range(0, len(shard_blocks), SHARDS_PER_BLOCK):
        group = shard_blocks[first : first + SHARDS_PER_BLOCK]
        blocks.append((group[0][0], group[-1][1]))
    return blocks


def fill_from_static_store(reader, rows, targets, leaves):
    series = reader.get_data_series(rows, 0, 1)[:, 0]  # (n_cells, variables of the store)
    for target in targets:
        leaves[target.leaf][:, target.leaf_channels] = target_values(series, target)


def fill_from_temporal_store(reader, rows, targets, leaves, start_day, end_day, num_workers):
    """Read the days start_day .. end_day - 1 of a store block by block into the leaves."""
    steps_per_day = int(pd.Timedelta("1D") / reader.time_grid.step)
    hours_per_step = 24 // steps_per_day
    first_step = start_day * steps_per_day
    blocks = time_blocks(reader, first_step, end_day * steps_per_day)

    def read_block(block):
        block_start, block_stop = block
        series = reader.get_data_series(rows, block_start, block_stop)
        first_hour = (block_start - first_step) * hours_per_step
        last_hour = (block_stop - first_step) * hours_per_step
        for target in targets:
            values = target_values(series, target)
            if hours_per_step > 1:  # a daily value holds for the 24 hours of its day
                values = np.repeat(values, hours_per_step, axis=1)
            leaves[target.leaf][:, first_hour:last_hour, target.leaf_channels] = values

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(read_block, blocks))  # the blocks write to separate hours of the leaves


def preload_modality(
    source, anchor, anchor_ids, paths, stats_zarr=None, anchor_grid=None, day_range=None, num_workers=8
):
    """Preload one modality for the anchors of a selection.

    Args:
        source: The modality's block of the sources config (stores, load, optional transform).
        anchor: "mtg", or the name of the modality whose grid rows the anchor ids are.
        anchor_ids: Anchor ids of the selection.
        paths: ExperimentPaths; gives the folder of every store.
        stats_zarr: Statistics file of the modality; None returns physical values.
        anchor_grid: LatLonGrid of the anchor modality; not needed for "mtg".
        day_range: (start_day, end_day) in days since 1970, end exclusive. Default: 2016 to 2025.
        num_workers: Threads that read time blocks in parallel.

    Returns:
        columns: {leaf: torch tensor}, one row per unique store cell.
        inverse: Row in the columns for every anchor. (n_anchors,)
        cell_lonlat: Lon, lat of the unique store cells. (n_cells, 2)
    """
    readers = open_readers(source, paths)
    grid = next(iter(readers.values())).spatial_grid
    rows = map_to_store_rows(anchor, anchor_ids, grid, anchor_grid)
    unique_rows, inverse = np.unique(rows, return_inverse=True)

    start_day, end_day = day_range if day_range is not None else (PRELOAD_START_DAY, PRELOAD_END_DAY)
    n_hours = (end_day - start_day) * 24
    leaves = allocate_leaves(source["load"], unique_rows.size, n_hours)
    for store_name, reader in readers.items():
        targets = leaf_targets(source, store_name, reader, stats_zarr)
        if reader.is_static:
            fill_from_static_store(reader, unique_rows, targets, leaves)
        else:
            fill_from_temporal_store(reader, unique_rows, targets, leaves, start_day, end_day, num_workers)

    columns = {leaf: torch.from_numpy(values) for leaf, values in leaves.items()}
    lat_index, lon_index = grid.rows_to_lat_lon_index(unique_rows)
    cell_lonlat = np.stack([grid.lon[lon_index], grid.lat[lat_index]], axis=1)
    return columns, inverse, cell_lonlat


def preload_modalities(
    sources, modality_ids, anchor, anchor_ids, paths, stats_spatial=None, day_range=None, num_workers=8
):
    """Preload several modalities for the anchors of a selection.

    Args:
        sources: The sources config, {modality: block}.
        modality_ids: Modalities to preload.
        stats_spatial: Selection whose statistics normalise the values; None returns physical values.
        anchor, anchor_ids, paths, day_range, num_workers: As in preload_modality.

    Returns:
        {"columns", "inverse", "var_names", "cell_lonlat"}, each a dict by modality; var_names
        holds the variable names of every leaf, in the order of its last axis.
    """
    anchor_grid = None
    if anchor != "mtg":
        anchor_readers = open_readers(sources[anchor], paths)
        anchor_grid = next(iter(anchor_readers.values())).spatial_grid

    columns, inverse, var_names, cell_lonlat = {}, {}, {}, {}
    for modality in modality_ids:
        source = sources[modality]
        stats_zarr = None if stats_spatial is None else paths.stats_zarr_path(modality, stats_spatial)
        columns[modality], inverse[modality], cell_lonlat[modality] = preload_modality(
            source, anchor, anchor_ids, paths, stats_zarr, anchor_grid, day_range, num_workers
        )
        var_names[modality] = {leaf: list(variables) for leaf, variables in source["load"].items()}
    return {"columns": columns, "inverse": inverse, "var_names": var_names, "cell_lonlat": cell_lonlat}
