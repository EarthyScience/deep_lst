import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentPaths:
    """Centralised path resolution for experiment artifacts."""

    experiment_root: Path = field(default_factory=lambda: Path("./experiments"))
    config_root: Path = field(default_factory=lambda: Path("./configs"))
    assets_root: Path = field(default_factory=lambda: Path("./assets"))
    plots_root: Path = field(default_factory=lambda: Path("./plots"))
    eozarr_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/weathergenerator/"))
    preload_cache_root: Path = field(default_factory=lambda: Path("/Net/Groups/BGI/tscratch/mwessel/tmp"))

    def experiment_dir(self, experiment: str) -> Path:
        return self.experiment_root / experiment

    def checkpoint_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_root / experiment / version / "checkpoints"

    def predictions_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_dir(experiment) / version / "results"

    def experiment_plots_dir(self, experiment: str, version: str) -> Path:
        return self.experiment_dir(experiment) / version / "plots"

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

    def statistics_path(self, modality, name) -> Path:
        return self.assets_root / modality / "statistics" / f"statistics_{name}.npz"

    def eozarr_store_path(self, store_name: str) -> Path:
        """Resolve a full eodata .eozarr store from its config-provided name.

        Falls back to the .zip sibling when the directory store is unusable. On 2026-08-22 the
        per-group zarr.json files were removed from era5_2014_2025.eozarr (8 of 9 groups), which
        makes every read fail with ArrayNotFoundError while the chunks themselves are also gone.
        The zip mirrors are intact, so prefer them rather than failing."""
        directory = self.eozarr_root / f"{store_name}.eozarr"
        archive = self.eozarr_root / f"{store_name}.eozarr.zip"
        directory_is_readable = (directory / "data" / "zarr.json").exists()
        if not directory_is_readable and archive.exists():
            return archive
        return directory

    def stats_zarr_path(self, modality: str, spatial: str) -> Path:
        """Zscore statistics zarr for *modality* (eodata ModalityNormalizer format),
        computed on the *spatial* pixel selection (indices_<spatial>.npz)."""
        return self.assets_root / "statistics" / f"{modality}_stats_{spatial}.zarr"

    def preload_cache_path(self, fingerprint: str, spatial: str | None = None) -> Path:
        """On-disk cache of the preloaded shared columns, keyed by a data-config fingerprint.

        The optional spatial name is a human-readable prefix (e.g. eval_passage, mtg); the
        fingerprint alone still guarantees correctness, the prefix is only for legibility.
        """
        self.preload_cache_root.mkdir(parents=True, exist_ok=True)
        name = f"preload_{spatial}_{fingerprint}.pt" if spatial else f"preload_{fingerprint}.pt"
        return self.preload_cache_root / name

    def spatial_indices_path(self, name: str) -> Path:
        """Anchor-id index file for a spatial selection. The ids are anchor-grid cell
        ids (MTG for training, seviri pixels for eval regions); every modality maps
        them to its own full-store rows at preload, so there is one file per selection."""
        return self.assets_root / "spatial" / f"indices_{name}.npz"

    def eval_indices_path(self, name: str) -> Path:
        """Resolve the dense evaluation pixel index file for *eval name*."""
        if name.startswith("train_"):
            return self.assets_root / "spatial" / f"indices_{name.removeprefix('train_')}.npz"
        return self.assets_root / "spatial" / f"indices_eval_{name}.npz"

    def eval_meta_path(self) -> Path:
        return self.assets_root / "spatial" / "eval_meta.json"

    @staticmethod
    def baseline_spatial(eval_name: str) -> str:
        """Spatial-selection name (indices_<name>.npz) whose baselines an eval name refers to:
        train_<sel> -> <sel>; bare region labels -> eval_<region>; eval_* passes through.
        A trailing init-stride token (_s3, _s7) names a thinned rerun of the same region
        and shares its baselines."""
        eval_name = re.sub(r"_s\d+$", "", eval_name)
        if eval_name.startswith("train_"):
            return eval_name.removeprefix("train_")
        return eval_name if eval_name.startswith("eval_") else f"eval_{eval_name}"

    def persistence_path(self, spatial: str) -> Path:
        return self.assets_root / "baselines" / "persistence" / f"{spatial}.nc"

    def baseline_path(self, spatial: str) -> Path:
        return self.assets_root / "baselines" / "era5" / f"{spatial}.nc"

    def modality_baseline_path(self, modality: str, spatial: str) -> Path:
        """Cross-modality reference baseline (e.g. era5_land skt) under its own modality folder."""
        return self.assets_root / "baselines" / modality / f"{spatial}.nc"

    def climatology_path(self, spatial: str, variant: str) -> Path:
        """variant : "hourly" | "smoothed_hourly" """
        return self.assets_root / "baselines" / f"climatology_{variant}" / f"{spatial}.nc"

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

    def results_plots_dir(self, experiment: str, version: str, eval_name: str) -> Path:
        d = self.results_dir(experiment, version, eval_name) / "plots"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def verification_dir(self, experiment: str, version: str, eval_name: str, kind: str) -> Path:
        """Climpred-verification artifacts next to the forecast store.
        kind: verif (raw mse+counts from the slurm scripts) | scores (derived ncs) | tables."""
        d = self.results_dir(experiment, version, eval_name) / kind
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


def stats_spatial(dataset_cfg: dict) -> str:
    """Spatial tag of the statistics a dataset normalizes with.

    This is the TRAINING pixel selection the stats zarrs were computed on. Eval regions
    override `spatial` per region but keep normalizing with the training statistics, so
    the tag lives in its own `stats_spatial` key (pinned to `${_spatial}` in train.yaml).
    Experiment configs saved before the key existed were all trained on mtg_new stats.
    """
    return dataset_cfg.get("stats_spatial", "mtg_new")


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
