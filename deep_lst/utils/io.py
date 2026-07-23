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
    stores_root: Path = field(
        default_factory=lambda: Path("/Net/Groups/BGI/tscratch/mwessel/wegen-stores/sampled_stores/")
    )
    eozarr_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/weathergenerator/"))
    preload_cache_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/mwessel/tmp"))
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

    def eozarr_store_path(self, store_name: str) -> Path:
        """Resolve a full eodata .eozarr store from its config-provided name."""
        return self.eozarr_root / f"{store_name}.eozarr"

    def stats_zarr_path(self, modality: str) -> Path:
        """Zscore statistics zarr for *modality* (eodata ModalityNormalizer format)."""
        return self.assets_root / "statistics" / f"{modality}_stats.zarr"

    def preload_cache_path(self, fingerprint: str) -> Path:
        """On-disk cache of the preloaded shared columns, keyed by a data-config fingerprint."""
        self.preload_cache_root.mkdir(parents=True, exist_ok=True)
        return self.preload_cache_root / f"preload_{fingerprint}.pt"

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
        if eval_name.startswith("train_"):
            return f"eval_{eval_name.removeprefix('train_')}"
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

    def tables_dir(self, experiment: str, version: str, kind: str) -> Path:
        """Evaluation tables grouped by family: kind in {scores, fc_limits, validity}."""
        d = self.results_root / "tables" / f"{experiment}_{version}" / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figures_dir(self, experiment: str, version: str) -> Path:
        """Per-model figure directory, parallel to the tables tree."""
        d = self.results_root / "plots" / f"{experiment}_{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_inc_variances(stats_zarr: Path) -> dict[str, float]:
    """Return {var: sigma_inc_norm^2} from a statistics zarr's `increments` group.

    The group stores the raw increment statistics (mean/std/min/max) for the prognostic
    variables, plus attrs `variables` (their names) and `channels` (their indices in the
    `data` leaf). The normalised increment variance is derived as
    (std_increment / std_var)^2 — the increment variance in z-score space, used to
    reweight per-variable loss contributions.

    Returns an empty dict if the zarr or the group is absent (modalities without
    prognostics fall through gracefully).
    """
    import zarr

    try:
        root = zarr.open_group(str(stats_zarr), mode="r")
        increments = root["increments"]
        data_std = root["data"]["std"][:]
    except (KeyError, FileNotFoundError):
        return {}
    increment_std = increments["std"][:]
    variables = increments.attrs["variables"]
    channels = increments.attrs["channels"]
    return {
        var: float((increment_std[i] / data_std[channel]) ** 2)
        for i, (var, channel) in enumerate(zip(variables, channels))
        if data_std[channel] > 0
    }


def load_computed_stats(stats_zarr: Path) -> dict[str, tuple[float, float]]:
    """Return {var: (mean, std)} from a statistics zarr's `computed` group.

    Statistics of the on-the-fly features (solar_zenith), written by
    compute_statistics.py over the statistics pixel/timestep selection.
    Returns an empty dict if the zarr or the group is absent.
    """
    import zarr

    try:
        group = zarr.open_group(str(stats_zarr), mode="r")["computed"]
    except (KeyError, FileNotFoundError):
        return {}
    means, stds = group["mean"][:], group["std"][:]
    variables = group.attrs["variables"]
    return {var: (float(means[i]), float(stds[i])) for i, var in enumerate(variables)}


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
