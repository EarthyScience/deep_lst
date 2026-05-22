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
    stores_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/mwessel/wegen-stores-new/"))
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
        """Return the most recent .ckpt file by alphabetical sort (last.ckpt wins)."""
        ckpt_dir = self.checkpoint_dir(experiment, version)
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        return ckpts[-1]

    def best_checkpoint(self, experiment: str, version: str) -> Path:
        """Return best.ckpt if it exists, otherwise fall back to latest_checkpoint."""
        best = self.checkpoint_dir(experiment, version) / "best.ckpt"
        if best.exists():
            return best
        return self.latest_checkpoint(experiment, version)

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
        return self.stores_root / f"{name}_long.zarr"

    def sampled_store_path(self, modality: str, spatial: str) -> Path:
        """Resolve the sampled zarr store for *modality* and *spatial* experiment name.

        e.g. sampled_store_path('seviri', 'KGLC_L') →
             .../seviri_sampled_KGLC_L.zarr
        """
        return self.stores_root / f"{modality}_sampled_{spatial}.zarr"

    def eval_indices_path(self, modality: str, name: str) -> Path:
        """Resolve the dense evaluation pixel index file for *modality* and *eval name*."""
        if name.startswith("train_"):
            return self.assets_root / modality / "spatial" / f"indices_{name.removeprefix('train_')}.npz"
        return self.assets_root / modality / "spatial" / f"indices_eval_{name}.npz"

    def eval_meta_path(self, modality: str) -> Path:
        return self.assets_root / modality / "spatial" / "eval_meta.json"

    @staticmethod
    def _eval_fname(eval_name: str) -> str:
        return eval_name if eval_name.startswith("eval_") else f"eval_{eval_name}"

    def persistence_path(self, eval_name: str) -> Path:
        return self.assets_root / "baselines" / "persistence" / f"{self._eval_fname(eval_name)}.nc"

    def baseline_path(self, eval_name: str) -> Path:
        return self.assets_root / "baselines" / "era5" / f"{self._eval_fname(eval_name)}.nc"

    def climatology_path(self, eval_name: str, variant: str) -> Path:
        """variant : "hourly" | "smoothed_hourly" """
        return self.assets_root / "baselines" / f"climatology_{variant}" / f"{self._eval_fname(eval_name)}.nc"

    def temporal_indices_path(self, split: str) -> Path:
        """Resolve the pre-computed temporal block index file for *split*.

        Expected location: assets/splits/time_indices_{split}.npz
        Key inside the file: ``time_indices`` (int64 T1H positions).
        """
        return self.assets_root / "temporal" / f"time_indices_{split}.npz"

    def metadata_path(self, experiment: str, version: str) -> Path:
        """Resolve the run-metadata JSON written at the start of each training run."""
        return self.experiment_dir(experiment) / version / "metadata.json"

    # --- results paths (separate from experiment dir, keyed by eval name) ---

    results_root: Path = field(default_factory=lambda: Path("./results"))

    def results_dir(self, experiment: str, version: str, eval_name: str) -> Path:
        return self.results_root / experiment / version / eval_name

    def forecast_path(self, experiment: str, version: str, eval_name: str, suffix: str = "") -> Path:
        path = self.results_dir(experiment, version, eval_name) / f"forecast{suffix}.zarr"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def metrics_path(self, experiment: str, version: str, eval_name: str, aggregation: str = "total") -> Path:
        """Aggregation: total | region_time | region_space | diurnal."""
        path = self.results_dir(experiment, version, eval_name) / f"metrics_{aggregation}.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def results_plots_dir(self, experiment: str, version: str, eval_name: str) -> Path:
        d = self.results_dir(experiment, version, eval_name) / "plots"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_inc_variances(path: Path) -> dict[str, float]:
    """Return {var: sigma_inc_norm^2} extracted from a statistics .npz file.

    For each "{var}_increment" entry in the file, computes the normalised
    increment variance: (std_increment / std_var)^2. This is the increment
    variance in z-score space, used to reweight per-variable loss contributions.

    Returns an empty dict if the file does not exist or contains no increment
    entries — old statistics files (pre-increment) fall through gracefully.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    stds = dict(zip(data["variables"], data["std"]))
    return {
        var[: -len("_increment")]: float((stds[var] / stds[var[: -len("_increment")]]) ** 2)
        for var in stds
        if var.endswith("_increment") and var[: -len("_increment")] in stds and stds[var[: -len("_increment")]] > 0
    }


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
