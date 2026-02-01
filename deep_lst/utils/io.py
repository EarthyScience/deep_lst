from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExperimentPaths:
    """Centralised path resolution for experiment artifacts."""

    experiment_root: Path = field(default_factory=lambda: Path("./experiments"))
    config_root: Path = field(default_factory=lambda: Path("./configs"))
    assets_root: Path = field(default_factory=lambda: Path("./assets"))
    stores_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/mwessel/wegen-stores/"))
    # --- directories ---

    def experiment_dir(self, experiment: str) -> Path:
        return self.experiment_root / experiment

    def checkpoint_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_root / experiment / version / "checkpoints"

    def profiler_dir(self, experiment: str) -> Path:
        return self.experiment_dir(experiment)

    def predictions_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_dir(experiment) / version / "results"

    def plots_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_dir(experiment) / version / "plots"

    # --- file resolutions ---

    def latest_checkpoint(self, experiment: str, version: str) -> Path:
        """Return the most recent .ckpt file by experiment sort."""
        ckpt_dir = self.checkpoint_dir(experiment, version)
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return ckpts[-1]

    def config_path(self, experiment: str) -> Path:
        return self.config_root / experiment / "config.yml"

    def scenes_path(self) -> Path:
        return self.config_root / "scenes.yml"

    def predictions_path(self, experiment: str, version: str) -> Path:
        path = self.predictions_dir(experiment, version) / "forecast.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plots_path(self, experiment: str, version: str, name: str) -> Path:
        path = self.plots_dir(experiment, version) / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --- asset data resolutions ---

    def statistics_path(self, modality, name) -> Path:
        return self.assets_root / modality / "statistics" / f"statistics_{name}.npz"

    def samples_path(self, name) -> Path:
        return self.assets_root / "samples" / f"{name}.npz"

    def store_path(self, name) -> Path:
        return self.stores_root / f"{name}.zarr"


def save_statistics_npz(statistics: dict, path: Path) -> None:
    """
    statistics: {var: {stat_name: scalar}}
    Writes one .npz with:
      - variables: array[str] of variable names (index)
      - each stat_name: array[float] aligned with `variables`
    """

    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    variables = np.array(list(statistics.keys()), dtype=object)

    # Collect all stat names (assumes scalars only)
    stat_names = sorted({k for stats in statistics.values() for k in stats.keys()})

    arrays: dict[str, np.ndarray] = {"variables": variables}

    for stat in stat_names:
        vals = []
        for var in variables:
            if stat not in statistics[var]:
                raise KeyError(f"Missing statistic '{stat}' for variable '{var}'.")
            vals.append(np.asarray(statistics[var][stat]).item())  # ensure scalar python value
        arrays[stat] = np.asarray(vals)

    np.savez(path, **arrays)
