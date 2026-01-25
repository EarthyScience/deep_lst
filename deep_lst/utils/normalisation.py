import pandas as pd


class Normaliser:
    def __init__(self, config, group: str):
        self.config = config
        self.statistics = pd.read_parquet(self.config["experiment_path"] + self.config[group]["statistics"])
        self._create_statistics_lookup()

    def _create_statistics_lookup(self):
        self.mean_lookup = self.statistics.set_index("variable")["mean"]
        self.std_lookup = self.statistics.set_index("variable")["std"]

    def normalise(self, data, var):
        return (data - self.mean_lookup[var]) / self.std_lookup[var]

    def denormalise(self, data, var):
        return data * self.std_lookup[var] + self.mean_lookup[var]
