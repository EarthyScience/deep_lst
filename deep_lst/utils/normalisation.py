import zarr
from eodata.util.misc import assemble_statistics_from_npz


class ZScoringNormalizer:
    def __init__(self, stats_path):
        self.stats = assemble_statistics_from_npz(stats_path)

    def normalise(self, data, var):
        return (data - self.stats[var]["mean"]) / self.stats[var]["std"]

    def denormalise(self, data, var):
        return data * self.stats[var]["std"] + self.stats[var]["mean"]


class StatsZarrNormalizer:
    """Per-variable z-score (de)normaliser backed by an eodata statistics zarr's `data` group.

    Same interface as ZScoringNormalizer, so the predict generators are untouched. Variable
    names come from the group's `variables` attr (written by compute_statistics.py).
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
