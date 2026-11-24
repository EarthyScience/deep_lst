import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from deep_lst.dataset.preload import PRELOAD_END_DAY, PRELOAD_START_DAY, preload_modalities
from deep_lst.stores import T1H
from deep_lst.utils import features
from deep_lst.utils.io import ExperimentPaths, load_computed_stats, stats_spatial

# Modality carrying the global time features (doy/hod); the finest / target grid.
_ANCHOR_MODALITY = "seviri"

PATHS = ExperimentPaths()

# time_indices_*.npz are hours since the data start (2016-01-01); the stores are indexed on the
# 1970 epoch, so a time index must be shifted by this offset to address the store.
# FLAGGED for later: cleaner to regenerate the indices as absolute T1H (see migration notes).
STORE_EPOCH_OFFSET = PRELOAD_START_DAY * 24  # 403224 hours = 16801 days


def load_samples(
    spatial_idx: np.ndarray,
    start_time: str | None = None,
    end_time: str | None = None,
    temporal_indices_path: str | list | None = None,
    horizon: int = 0,
    lookback: int = 0,
    init_hours: list[int] | None = None,
) -> np.ndarray:
    if temporal_indices_path is not None:
        time_idx = np.load(temporal_indices_path)["time_indices"].astype(np.int64) + STORE_EPOCH_OFFSET
        axis_end_hour = PRELOAD_END_DAY * 24
        time_idx = time_idx[time_idx >= STORE_EPOCH_OFFSET + lookback]
        time_idx = time_idx[time_idx + horizon < axis_end_hour]
    else:
        time_idx = np.arange(
            start=T1H.date_to_index(start_time) + lookback,
            stop=T1H.date_to_index(end_time) - horizon,
        )

    if init_hours is not None:
        time_idx = time_idx[np.isin(time_idx % 24, init_hours)]

    time, space = np.meshgrid(time_idx, spatial_idx, indexing="ij")
    indices = np.stack([space.ravel(), time.ravel()], axis=1)  # (N_time * N_spatial, 2)
    return indices


def _load_anchor_ids(spatial_name, paths=None):
    """Anchor cell ids for a split: global MTG ids for training, seviri equiangular pixels
    for an eval holdout region."""
    paths = paths or PATHS
    return np.sort(np.load(paths.spatial_indices_path(spatial_name))["src_indices"].astype(np.int64))


def _window_leaves(columns, row, hour, lookback, horizon):
    """Slice one (pixel row, anchor hour) window out of a preloaded leaf dict.

    Temporal leaves (3-D cache) are sliced [hour - L : hour + H + 1]; static leaves
    (2-D cache) are broadcast to the same length. Returns a TensorDict of (L + H + 1, C).
    """
    length = lookback + horizon + 1
    local_hour = hour - STORE_EPOCH_OFFSET
    lo, hi = local_hour - lookback, local_hour + horizon + 1
    window = TensorDict({}, batch_size=[])
    for key, column in columns.items():
        if column.ndim == 2:  # static: (n_pixels, C) -> broadcast over the window
            window[key] = column[row].unsqueeze(0).expand(length, -1)
        else:  # temporal: (n_pixels, n_hours, C) -> contiguous window slice
            window[key] = column[row, lo:hi]
    return window


def _to_named_channels(leaves, var_names):
    """Explode preloaded leaf windows into per-variable (L+H+1, 1) named channels (option A).

    var_names maps each loaded leaf to its ordered variable names (the config `load`), matching
    the channel order the reader returned. The collator then packs by (modality, var) name.
    """
    channels = TensorDict({}, batch_size=[])
    for leaf, names in var_names.items():
        window = leaves[leaf]
        for channel, name in enumerate(names):
            # (L+H+1, 1, 1, 1): keep the singleton H, W spatial dims the packed model expects
            channels[name] = window[:, channel : channel + 1].unsqueeze(1).unsqueeze(1)
    return channels


class MultimodalDataset(Dataset):
    def __init__(
        self,
        columns,
        inverse,
        var_names,
        cell_lonlat,
        indices,
        n_spatial,
        lookback,
        horizon,
        computed_stats,
        quality_mask,
    ):
        """
        columns: {modality: {leaf_key: (n_unique, n_hours, C) | (n_unique, C)}} preloaded in RAM,
            NaN where missing; deduped to the unique native cells (shared across splits).
        inverse: {modality: (n_spatial,)} anchor position -> row into that modality's cache
            (coarse modalities map many anchors to one native cell).
        var_names: {modality: {leaf_key: [var names]}} — the config `load`, for named channels.
        cell_lonlat: {modality: (n_unique, 2)} lon/lat per cached cell (for solar_zenith).
        indices: (N, 2) array of (anchor_id, anchor_hour), time-major.
        lookback / horizon: {modality: int} window extents per modality.
        quality_mask: {modality: {field, se_max, target} | None} — NaN target where field > se_max.
        """
        self.columns = columns
        self.inverse = inverse
        self.var_names = var_names
        self.cell_lonlat = cell_lonlat
        self.indices = indices
        self.n_spatial = n_spatial
        self.lookback = lookback
        self.horizon = horizon
        self.samples = len(self.indices)
        self.computed_stats = computed_stats
        self.quality_mask = quality_mask
        print(
            f"Initialized MultimodalDataset with {self.samples} samples "
            f"({n_spatial} anchors × {self.samples // n_spatial} times, time-major)."
        )

    def __len__(self):
        return self.samples

    def _add_computed(self, channels, modality, lonlat, hour, lookback, horizon):
        """Attach computed features: solar_zenith + hod per grid; doy only on the anchor modality.

        Only solar_zenith is standardised; the cyclic cos/sin encodings stay raw in [-1, 1].
        """
        hours = np.arange(hour - lookback, hour + horizon + 1)
        solar = features.solar_features(float(lonlat[1]), float(lonlat[0]), hours)
        mean, std = self.computed_stats[modality]["solar_zenith"]
        solar["solar_zenith"] = (solar["solar_zenith"] - mean) / std
        for name, values in solar.items():
            channels[name] = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1)
        if modality == _ANCHOR_MODALITY:
            for name, values in features.time_features(hours).items():
                channels[name] = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1, 1, 1)

    def _apply_quality_mask(self, channels, modality):
        """NaN the target var where its quality field exceeds se_max (retrieval-error threshold).

        Cloned before writing — the channel tensors are views into the shared preload cache.
        """
        spec = self.quality_mask.get(modality)
        if not spec:
            return
        field = channels[spec["field"]]  # NaN where the retrieval is invalid → comparison is False
        target = channels[spec["target"]].clone()
        target[field > spec["se_max"]] = float("nan")
        channels[spec["target"]] = target

    def __getitem__(self, idx):
        pos = idx % self.n_spatial
        anchor_id, hour = self.indices[idx]

        data = TensorDict({}, batch_size=[])
        for modality, columns in self.columns.items():
            row = int(self.inverse[modality][pos])
            lookback, horizon = self.lookback[modality], self.horizon[modality]
            leaves = _window_leaves(columns, row, hour, lookback, horizon)
            channels = _to_named_channels(leaves, self.var_names[modality])
            self._add_computed(channels, modality, self.cell_lonlat[modality][row], hour, lookback, horizon)
            self._apply_quality_mask(channels, modality)
            data[modality] = channels

        data["spatial_idx"] = torch.tensor(anchor_id, dtype=torch.long)
        data["time_idx"] = torch.tensor(hour, dtype=torch.long)
        return data

    @property
    def time_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 1])

    @property
    def spatial_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 0])

    def pixel_lonlat(self) -> np.ndarray:
        """(n_spatial, 2) lon/lat per anchor pixel, from the target modality's cached cells."""
        return self.cell_lonlat[_ANCHOR_MODALITY][self.inverse[_ANCHOR_MODALITY]]

    def non_land_mask(self, classes=(17, 18)) -> np.ndarray:
        """(n_spatial,) bool; True where the anchor pixel's LANDCOV is ocean (17) or water (18)."""
        categorical = self.var_names[_ANCHOR_MODALITY]["data_static_categorical"]
        channel = categorical.index("LANDCOV")
        landcov = self.columns[_ANCHOR_MODALITY]["data_static_categorical"][:, channel].numpy()
        return np.isin(landcov[self.inverse[_ANCHOR_MODALITY]], classes)

    @classmethod
    def preload_shared(cls, modality_ids, paths=None, **kwargs):
        """Preload each modality's native columns once, shared across train/val/test splits.

        Every anchor cell (MTG for training, seviri pixels for eval) is mapped to the nearest
        cell of each modality's stores; coarse modalities are deduped so only unique cells are
        read (deep_lst/dataset/preload.py). Columns depend only on (anchors, stores), not the
        temporal split, so this runs once — a full-axis cache per split would not fit in RAM.
        """
        paths = paths or PATHS
        spatial_name = kwargs["spatial"]
        stats_tag = stats_spatial(kwargs)
        anchor_ids = _load_anchor_ids(spatial_name, paths)
        anchor = kwargs.get("anchor", "mtg")
        num_workers = kwargs.get("preload_workers", 8)
        sources = {name: value for name, value in kwargs.items() if isinstance(value, dict) and "stores" in value}

        # Disk cache: the preload is deterministic given the anchors, stores, load specs and
        # statistics, so later runs (restarts, HP/architecture sweeps) skip the store reads.
        # Model-role changes reuse the cache; editing a `stores:` or `load:` section or
        # regenerating a statistics file changes the fingerprint and triggers a rebuild.
        fingerprint_spec = {
            "anchors": hashlib.sha256(np.ascontiguousarray(anchor_ids).tobytes()).hexdigest(),
            "modalities": {
                modality: {
                    "stores": sources[modality]["stores"],
                    "load": sources[modality]["load"],
                    "stats_mtime": os.path.getmtime(paths.stats_zarr_path(modality, stats_tag) / "zarr.json"),
                    "axis": [PRELOAD_START_DAY, PRELOAD_END_DAY],
                }
                for modality in modality_ids
            },
        }
        for modality in modality_ids:
            store_roots = {}
            for store_name in sources[modality]["stores"]:
                store_root = paths.store_roots.get(store_name)
                if store_root is not None:  # only when set, so the key does not depend on the common root
                    store_roots[store_name] = str(Path(store_root).resolve())
            if store_roots:
                fingerprint_spec["modalities"][modality]["store_roots"] = store_roots
            transform = sources[modality].get("transform")
            if transform is not None:  # only when set, as for store_roots
                fingerprint_spec["modalities"][modality]["transform"] = transform
        fingerprint = hashlib.sha256(json.dumps(fingerprint_spec, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = paths.preload_cache_path(fingerprint, spatial_name)
        if cache_path.exists():
            print(f"Loading preload cache {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        shared = preload_modalities(
            sources, modality_ids, anchor, anchor_ids, paths, stats_spatial=stats_tag, num_workers=num_workers
        )
        shared["anchor_ids"] = anchor_ids
        shared["spatial"] = spatial_name
        print(f"Saving preload cache {cache_path}")
        temporary_path = cache_path.with_suffix(".tmp")
        torch.save(shared, temporary_path)
        temporary_path.rename(cache_path)  # atomic-ish: never leaves a half-written cache behind
        return shared

    @classmethod
    def from_modalities(cls, modality_ids, shared, paths=None, **kwargs):
        """Build one split's dataset from the shared preloaded columns + this split's anchors."""
        paths = paths or PATHS
        stats_tag = stats_spatial(kwargs)
        computed_stats = {m: load_computed_stats(paths.stats_zarr_path(m, stats_tag)) for m in modality_ids}

        if kwargs["spatial"] != shared["spatial"]:
            raise ValueError(f"Shared cache built for spatial={shared['spatial']!r}, got {kwargs['spatial']!r}")

        temporal_split = kwargs["temporal"]
        if temporal_split is not None:
            temporal_paths = paths.temporal_indices_path(temporal_split)
        else:
            temporal_paths = None

        lookback = {m: kwargs[m].get("lookback", 0) for m in modality_ids}
        horizon = {m: kwargs[m].get("horizon", 0) for m in modality_ids}
        quality_mask = {m: kwargs[m].get("quality_mask") for m in modality_ids}
        # Clip anchors by the strictest extent so every modality can serve its window.
        indices = load_samples(
            shared["anchor_ids"],
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            temporal_indices_path=temporal_paths,
            horizon=max(horizon.values()),
            lookback=max(lookback.values()),
            init_hours=kwargs.get("init_hours"),
        )
        n_spatial = len(shared["anchor_ids"])
        return cls(
            shared["columns"],
            shared["inverse"],
            shared["var_names"],
            shared["cell_lonlat"],
            indices,
            n_spatial,
            lookback,
            horizon,
            computed_stats,
            quality_mask,
        )
