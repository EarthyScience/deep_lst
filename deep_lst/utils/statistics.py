"""Statistics files: mean, standard deviation, minimum and maximum of every variable.

A statistics file (assets/statistics/<modality>_stats_<selection>.zarr) is a zarr v3 group with
one group per kind of variable:

    data, data_daily, data_static   variables read from a store
    computed                        features computed at run time (solar_zenith)
    increments                      change of a variable from one hour to the next

Every group holds the float64 arrays mean, std, min and max of shape (n_variables,) and the
attribute `variables` with the names in the order of the arrays. The group `increments` also
has the attribute `channels`: the position of each of its variables in the group `data`. The
root attribute `transform` names the value transform the statistics were computed under; it is
absent when there is none.

This is the layout that eodata's ModalityStatistics wrote, so files written here and files
written by the eodata pipeline are read by the same code.
"""

import numpy as np
import zarr

VALUE_GROUPS = ("data", "data_daily", "data_static")  # groups of variables read from a store
STATISTIC_NAMES = ("mean", "std", "min", "max")


def variable_statistics(values):
    """Mean, standard deviation, minimum and maximum of every variable, ignoring NaN.

    Args:
        values: Array (..., n_variables); the statistics are taken over all other axes.

    Returns:
        {"mean", "std", "min", "max"}: float64 arrays (n_variables,). The standard deviation is
        the sample standard deviation (divided by n - 1), as in the eodata pipeline.
    """
    values = np.asarray(values)
    n_variables = values.shape[-1]
    statistics = {name: np.empty(n_variables, dtype=np.float64) for name in STATISTIC_NAMES}
    for channel in range(n_variables):
        column = values[..., channel].astype(np.float64).ravel()
        statistics["mean"][channel] = np.nanmean(column)
        statistics["std"][channel] = np.nanstd(column, ddof=1)
        statistics["min"][channel] = np.nanmin(column)
        statistics["max"][channel] = np.nanmax(column)
    return statistics


def increment_statistics(values):
    """Statistics of the change of every variable from one time step to the next.

    Args:
        values: Array (..., n_steps, n_variables) of consecutive time steps. A change is only
            taken within the second-to-last axis, so (cells, days, 24, n_variables) gives the
            hourly changes within each day.

    Returns:
        As variable_statistics.
    """
    increments = np.diff(np.asarray(values), axis=-2)
    return variable_statistics(increments)


def write_statistics(stats_zarr, groups, transform=None):
    """Write a statistics file.

    Args:
        stats_zarr: Folder of the file; an existing file is replaced.
        groups: {group name: {"variables": names, "mean": ..., "std": ..., "min": ..., "max": ...}};
            the group "increments" also needs "channels".
        transform: Name of the value transform the statistics were computed under, or None.
    """
    attributes = {} if transform is None else {"transform": transform}
    root = zarr.open_group(str(stats_zarr), mode="w", zarr_format=3, attributes=attributes)
    for group_name, content in groups.items():
        group_attributes = {"dimension_names": ["channel"], "variables": list(content["variables"])}
        if "channels" in content:
            group_attributes["channels"] = [int(channel) for channel in content["channels"]]
        group = root.create_group(group_name, attributes=group_attributes)
        for statistic in STATISTIC_NAMES:
            values = np.atleast_1d(np.asarray(content[statistic], dtype=np.float64))
            array = group.create_array(
                statistic, shape=values.shape, dtype="float64", chunks=values.shape, compressors=None, fill_value=0.0
            )
            array[:] = values


def read_variable_statistics(stats_zarr):
    """Mean and standard deviation of every variable read from a store.

    Returns:
        {variable: (mean, std)} over the groups data, data_daily and data_static.
    """
    root = zarr.open_group(str(stats_zarr), mode="r")
    statistics = {}
    for group_name in VALUE_GROUPS:
        if group_name not in root:
            continue
        group = root[group_name]
        means = group["mean"][:]
        stds = group["std"][:]
        for position, variable in enumerate(group.attrs["variables"]):
            statistics[variable] = (float(means[position]), float(stds[position]))
    return statistics
