import numpy as np
import zarr

from deep_lst.utils.statistics import read_variable_statistics

STD_FLOOR = 1e-10  # a standard deviation below this is replaced by it, as in the eodata pipeline


def signed_log1p(values):
    """sign(x) * log(1 + |x|): compresses heavy tails, keeps zero at zero and the sign."""
    return np.sign(values) * np.log1p(np.abs(values))


def signed_expm1(values):
    """Inverse of signed_log1p."""
    return np.sign(values) * np.expm1(np.abs(values))


# name of a value transform -> (transform, its inverse)
TRANSFORMS = {"signed_log1p": (signed_log1p, signed_expm1)}


class VariableNormalizer:
    """Z-score of several variables at once, from a statistics file.

    Normalises arrays whose last axis runs over `variables`, as the readers of deep_lst.stores
    return them. With a transform, the values are transformed first and the statistics must have
    been computed on transformed values. Reproduces the normalisation of the eodata pipeline:
    float32, with the standard deviation floored at STD_FLOOR.

    Attributes:
        variables: Names of the variables, in the order of the last axis.
        mean: Their means. float32 (n_variables,)
        std: Their standard deviations. float32 (n_variables,)
        transform: Name of the value transform, or None.
    """

    def __init__(self, stats_zarr, variables, transform=None):
        if transform is not None and transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform {transform!r}, expected one of {sorted(TRANSFORMS)}")
        statistics = read_variable_statistics(stats_zarr)
        self.variables = list(variables)
        means = [statistics[variable][0] for variable in self.variables]
        stds = [max(statistics[variable][1], STD_FLOOR) for variable in self.variables]
        self.mean = np.array(means, dtype=np.float32)
        self.std = np.array(stds, dtype=np.float32)
        self.transform = transform

    def normalise(self, values):
        """Physical values (..., n_variables) -> normalised float32 values of the same shape."""
        values = np.asarray(values, dtype=np.float32)
        if self.transform is not None:
            forward, _ = TRANSFORMS[self.transform]
            values = forward(values)
        return (values - self.mean) / self.std

    def denormalise(self, values):
        """Normalised values (..., n_variables) -> physical float32 values of the same shape."""
        values = np.asarray(values, dtype=np.float32) * self.std + self.mean
        if self.transform is not None:
            _, inverse = TRANSFORMS[self.transform]
            values = inverse(values)
        return values


class StatsZarrNormalizer:
    """Per-variable z-score (de)normaliser backed by an eodata statistics zarr's `data` group.

    Variable names come from the group's `variables` attr (written by compute_statistics.py).
    """

    def __init__(self, stats_zarr):
        group = zarr.open_group(str(stats_zarr), mode="r")["data"]
        means = group["mean"][:]
        stds = group["std"][:]
        variables = group.attrs["variables"]
        self.stats = {var: {"mean": float(means[i]), "std": float(stds[i])} for i, var in enumerate(variables)}

    def normalise(self, data, var):
        return (data - self.stats[var]["mean"]) / self.stats[var]["std"]

    def denormalise(self, data, var):
        return data * self.stats[var]["std"] + self.stats[var]["mean"]
