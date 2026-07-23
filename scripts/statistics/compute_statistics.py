"""Recompute statistics for one modality over the MTG training pixels + statistics timesteps.

Base statistics (raw mean/std/min/max per store leaf) mirror eodata's cli/stats.py — WelfordMetric
per continuous leaf -> ModalityStatistics.to_zarr — but sourced from deep_lst's raw streaming
UnimodalDataset (no normalization). Prognostic increment statistics (a deep_lst modelling concept)
are stored as normalised increment variances in an `increments` group inside the same zarr.

One modality per run (submit per-modality jobs in parallel):
    uv run scripts/statistics/compute_statistics.py modality=seviri
"""

import hydra
import numpy as np
import torch
import zarr
from eodata.statistics import ArrayStatistics, ModalityStatistics, WelfordMetric
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_lst.dataset.dataset import _LEAF_ARRAY, UnimodalDataset
from deep_lst.utils import features
from deep_lst.utils.io import ExperimentPaths
from deep_lst.utils.spatial import store_grid

PATHS = ExperimentPaths()

# On-the-fly features that get statistics. The doy/hod cos/sin encodings are bounded in
# [-1, 1] by construction and stay raw — z-scoring the components of a cyclic pair
# separately would distort the encoding.
COMPUTED_VARIABLES = ["solar_zenith"]


def _welford(device: torch.device) -> WelfordMetric:
    return WelfordMetric(dim=(0,)).to(device=device).to(dtype=torch.float64)


def _flatten_channels(values: torch.Tensor) -> torch.Tensor:
    """(B, days, C, 24) or (B, days, C) -> (N, C) with the channel axis last."""
    if values.ndim == 4:  # temporal leaf: move the channel axis to the end
        values = values.movedim(2, -1)  # (B, days, 24, C)
    return values.reshape(-1, values.shape[-1])


@torch.no_grad()
def compute_statistics(dataset, prognostic_channels, device, num_workers):
    """Base per-leaf statistics + optional prognostic increment statistics (NaN ignored)."""
    loader = DataLoader(dataset, batch_size=8, num_workers=num_workers, drop_last=False)
    base: dict[str, WelfordMetric] = {}
    increment = _welford(device) if prognostic_channels else None

    for batch in tqdm(loader):
        for leaf, values in batch.items():
            values = values.to(device)
            base.setdefault(leaf, _welford(device)).update(_flatten_channels(values).to(torch.float64))
        if increment is not None:
            data = batch["data"].to(device)  # (B, days, C, 24)
            diff = (data[..., 1:] - data[..., :-1])[:, :, prognostic_channels, :]  # within-day 1-hour increments
            increment.update(diff.movedim(2, -1).reshape(-1, len(prognostic_channels)).to(torch.float64))

    base_stats = {leaf: metric.compute() for leaf, metric in base.items()}
    increment_stats = increment.compute() if increment is not None else None
    return base_stats, increment_stats


def computed_feature_statistics(rows, stat_days, modality) -> ArrayStatistics:
    """Statistics of the computed features over the same pixels × hours as the base stats.

    Computed features are pure functions of (cell lon/lat, hour) — no store reads — so they
    are exactly reproducible from the statistics pixel/timestep selection alone.
    """
    centers = store_grid(modality).centers[np.asarray(rows)]  # (n_rows, 2) lon/lat
    hours = (np.asarray(stat_days)[:, None] * 24 + np.arange(24)).ravel()
    zenith = np.stack([features.solar_zenith(float(lat), float(lon), hours) for lon, lat in centers])
    values = zenith.astype(np.float64)
    stats = {
        "mean": np.atleast_1d(values.mean()),
        "std": np.atleast_1d(values.std()),
        "min": np.atleast_1d(values.min()),
        "max": np.atleast_1d(values.max()),
    }
    return ArrayStatistics.from_dict(stats, dimension_names=["channel"])


@hydra.main(config_path="../../conf", config_name="statistics", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    modality = config["modality"]
    prognostics = config[modality].get("prognostics", [])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = UnimodalDataset.from_config(
        modality=modality,
        store=config[modality]["store"],
        spatial=config["spatial"],
        anchor=config["anchor"],
        temporal=config["temporal"],
    )
    data_fields = list(dataset.reader.data_arr.dtype.fields)
    prognostic_channels = [data_fields.index(var) for var in prognostics]
    print(f"modality={modality}  pixels={len(dataset.rows)}  days={len(dataset.stat_days)}  prognostics={prognostics}")

    base_stats, increment_stats = compute_statistics(dataset, prognostic_channels, device, config["num_workers"])

    # Base statistics per leaf, plus (if any) the prognostic increments — all stored as the same
    # ArrayStatistics (mean/std/min/max). load_inc_variances derives (std_increment/std_var)^2 later.
    def _array_stats(stats):
        # atleast_1d: single-variable leaves (e.g. seviri data_static=DEM) squeeze to 0-D in the
        # metric; store them as (1,) so the zarr matches the normalizer's per-channel layout.
        arrays = {k: np.atleast_1d(v.cpu().numpy()) for k, v in stats.items()}
        return ArrayStatistics.from_dict(arrays, dimension_names=["channel"])

    modality_stats = {leaf: _array_stats(stats) for leaf, stats in base_stats.items()}
    modality_stats["computed"] = computed_feature_statistics(dataset.rows, dataset.stat_days, modality)
    if prognostics:
        modality_stats["increments"] = _array_stats(increment_stats)

    out_path = PATHS.stats_zarr_path(modality)
    ModalityStatistics(modality_stats).to_zarr(str(out_path))

    # Record variable names next to the positional statistics: base leaves carry the full
    # store variable order, the computed group its feature names.
    root = zarr.open_group(str(out_path), mode="a")
    for leaf in base_stats:
        root[leaf].attrs["variables"] = list(getattr(dataset.reader, _LEAF_ARRAY[leaf]).dtype.fields)
    root["computed"].attrs["variables"] = list(COMPUTED_VARIABLES)

    if prognostics:  # record which data variables the increment channels correspond to
        group = root["increments"]
        group.attrs["variables"] = list(prognostics)
        group.attrs["channels"] = [int(channel) for channel in prognostic_channels]

    for leaf, stats in base_stats.items():
        print(
            f"  {leaf}: mean={np.round(stats['mean'].cpu().numpy(), 3)}  std={np.round(stats['std'].cpu().numpy(), 3)}"
        )
    if prognostics:
        print(f"  increments std={np.round(increment_stats['std'].cpu().numpy(), 3)} for {prognostics}")
    print(f"saved statistics -> {out_path}")


if __name__ == "__main__":
    main()
