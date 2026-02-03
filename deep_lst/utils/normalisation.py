from wedata.util.misc import assemble_statistics_from_npz


class ZScoringNormalizer:
    def __init__(self, stats_path):
        self.stats = assemble_statistics_from_npz(stats_path)

    def normalise(self, data, var):
        return (data - self.stats[var]["mean"]) / self.stats[var]["std"]

    def denormalise(self, data, var):
        return data * self.stats[var]["std"] + self.stats[var]["mean"]
