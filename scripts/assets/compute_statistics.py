"""Compute the statistics of one modality over a pixel selection and the statistics days.

Reads the physical values of every variable in the modality's `load` for the unique store cells
of the selection (spatial) on the days of the statistics split (temporal), and writes
assets/statistics/<modality>_stats_<spatial>.zarr (layout: deep_lst/utils/statistics.py):

    data, data_daily, data_static   mean, std, min, max of the variables of these leaves
    increments                      the same for the hour-to-hour change of `increment_vars`
    computed                        the same for the computed feature solar_zenith

With a `transform` on the modality, the statistics of the leaf `data` and of the increments are
those of the transformed values, which is what the normaliser z-scores. Class variables and
metadata get no statistics.

One modality per run:
    python scripts/assets/compute_statistics.py modality=seviri spatial=<name>
"""

from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from deep_lst.dataset.dataset import STORE_EPOCH_OFFSET, _load_anchor_ids
from deep_lst.dataset.preload import NORMALISED_LEAVES, STATIC_LEAVES, open_readers, preload_modality, variable_stores
from deep_lst.utils import features
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.normalisation import TRANSFORMS
from deep_lst.utils.statistics import increment_statistics, variable_statistics, write_statistics

# On-the-fly features that get statistics. The doy/hod cos/sin encodings are bounded in
# [-1, 1] by construction and stay raw — z-scoring the components of a cyclic pair
# separately would distort the encoding.
COMPUTED_VARIABLES = ["solar_zenith"]
HOURS_PER_DAY = 24


def statistics_days(paths, temporal):
    """Days of the statistics split, in days since 1970: every day with an hour in the split."""
    hours = np.load(paths.temporal_indices_path(temporal))["time_indices"].astype(np.int64)
    return np.unique((hours + STORE_EPOCH_OFFSET) // HOURS_PER_DAY)


def daily_leaves(source, paths):
    """Leaves of `load` whose variables come from a daily store: their 24 hourly values are one value."""
    readers = open_readers(source, paths)
    assignment = variable_stores(source)
    daily = []
    for leaf, variables in source["load"].items():
        reader = readers[assignment[variables[0]]]
        if not reader.is_static and reader.time_grid.name == "T1D":
            daily.append(leaf)
    return daily


def read_days(source, anchor, anchor_ids, days, paths, anchor_grid, num_workers):
    """Physical values of every leaf on the given days.

    Returns:
        leaves: {leaf: array}; (n_cells, n_days, 24, n_variables) for the leaves on the hourly
            axis, (n_cells, n_variables) for the static ones.
        cell_lonlat: Lon, lat of the unique store cells. (n_cells, 2)
    """

    def read_day(day):
        day_range = (int(day), int(day) + 1)
        return preload_modality(
            source, anchor, anchor_ids, paths, anchor_grid=anchor_grid, day_range=day_range, num_workers=1
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        per_day = list(pool.map(read_day, days))

    first_columns, _, cell_lonlat = per_day[0]
    leaves = {}
    for leaf in source["load"]:
        if leaf in STATIC_LEAVES:
            leaves[leaf] = first_columns[leaf].numpy()
        else:
            day_values = [columns[leaf].numpy() for columns, _, _ in per_day]
            leaves[leaf] = np.stack(day_values, axis=1)
    return leaves, cell_lonlat


def solar_zenith_statistics(cell_lonlat, days):
    """Statistics of solar_zenith over the cells and the hours of the days.

    Computed features are pure functions of (cell lon/lat, hour), no store reads.
    """
    hours = (np.asarray(days)[:, None] * HOURS_PER_DAY + np.arange(HOURS_PER_DAY)).ravel()
    zenith = [features.solar_features(float(lat), float(lon), hours)["solar_zenith"] for lon, lat in cell_lonlat]
    values = np.stack(zenith).astype(np.float64)
    return variable_statistics(values[..., None])


def modality_statistics(source, leaves, cell_lonlat, days, daily):
    """The groups of a statistics file for one modality.

    Args:
        source: The modality's block of the sources config.
        leaves, cell_lonlat: As returned by read_days.
        days: The days the leaves hold.
        daily: Leaves that hold daily values repeated over the hours of a day.
    """
    load = source["load"]
    transform = source.get("transform")
    increment_vars = source.get("increment_vars", [])
    groups = {}
    for leaf in NORMALISED_LEAVES:
        if leaf not in load:
            continue
        values = leaves[leaf]
        if leaf in daily:
            values = values[:, :, :1]  # one value per day, not 24 copies of it
        if leaf == "data" and transform is not None:
            forward, _ = TRANSFORMS[transform]
            values = forward(values)
        groups[leaf] = dict(variable_statistics(values), variables=list(load[leaf]))
        if leaf == "data" and increment_vars:
            channels = [load[leaf].index(variable) for variable in increment_vars]
            increments = increment_statistics(values[..., channels])  # hourly changes within each day
            groups["increments"] = dict(increments, variables=list(increment_vars), channels=channels)
    groups["computed"] = dict(solar_zenith_statistics(cell_lonlat, days), variables=list(COMPUTED_VARIABLES))
    return groups


@hydra.main(config_path="../../conf", config_name="statistics", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    paths = ExperimentPaths.from_config(config.get("paths"))
    modality = config["modality"]
    sources = config["sources"]  # the same definition train.yaml composes
    source = sources[modality]
    anchor = config["anchor"]

    anchor_ids = _load_anchor_ids(config["spatial"], paths)
    days = statistics_days(paths, config["temporal"])
    anchor_grid = None
    if anchor != "mtg":
        anchor_readers = open_readers(sources[anchor], paths)
        anchor_grid = next(iter(anchor_readers.values())).spatial_grid
    print(
        f"modality={modality}  anchors={len(anchor_ids)}  days={len(days)}  "
        f"increment_vars={source.get('increment_vars', [])}  transform={source.get('transform')}"
    )

    leaves, cell_lonlat = read_days(source, anchor, anchor_ids, days, paths, anchor_grid, config["num_workers"])
    groups = modality_statistics(source, leaves, cell_lonlat, days, daily_leaves(source, paths))

    out_path = paths.stats_zarr_path(modality, config["spatial"])
    write_statistics(out_path, groups, transform=source.get("transform"))
    for name, group in groups.items():
        print(f"  {name} {group['variables']}: mean={np.round(group['mean'], 3)}  std={np.round(group['std'], 3)}")
    print(f"saved statistics -> {out_path}")


if __name__ == "__main__":
    main()
