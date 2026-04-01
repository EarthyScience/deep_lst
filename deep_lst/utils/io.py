from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExperimentPaths:
    """Centralised path resolution for experiment artifacts."""

    experiment_root: Path = field(default_factory=lambda: Path("./experiments"))
    config_root: Path = field(default_factory=lambda: Path("./configs"))
    assets_root: Path = field(default_factory=lambda: Path("./assets"))
    plots_root: Path = field(default_factory=lambda: Path("./plots"))
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

    def experiment_plots_dir(self, experiment: str, version: str) -> Path:
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

    def scenes_path(self, experiment) -> Path:
        return self.config_root / f"{experiment}.yml"

    def predictions_path(self, experiment: str, version: str, suffix: str = "") -> Path:
        path = self.predictions_dir(experiment, version) / f"forecast{suffix}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def experiment_plots_path(self, experiment: str, version: str, name: str) -> Path:
        path = self.experiment_plots_dir(experiment, version) / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def plots_path(self, subpath: str, name: str) -> Path:
        path = self.plots_root / subpath / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --- asset data resolutions ---

    def statistics_path(self, modality, name) -> Path:
        return self.assets_root / modality / "statistics" / f"statistics_{name}.npz"

    def spatial_indices_path(self, modality: str, name: str) -> Path:
        return self.assets_root / modality / "spatial" / f"indices_{name}.npz"

    def store_path(self, name) -> Path:
        return self.stores_root / f"{name}.zarr"

    def eval_indices_path(self, modality: str, name: str) -> Path:
        """
        Resolve the dense evaluation pixel index file for *modality* and *eval name*.
        """
        return self.assets_root / modality / "eval" / f"indices_{name}.npz"

    def temporal_indices_path(self, split: str) -> Path:
        """Resolve the pre-computed temporal block index file for *split*.

        Expected location: assets/splits/time_indices_{split}.npz
        Key inside the file: ``time_indices`` (int64 T1H positions).
        """
        return self.assets_root / "temporal" / f"time_indices_{split}.npz"

    def metadata_path(self, experiment: str, version: str) -> Path:
        """Resolve the run-metadata JSON written at the start of each training run."""
        return self.experiment_dir(experiment) / version / "metadata.json"


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
