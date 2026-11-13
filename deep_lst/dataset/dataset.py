import hashlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import eodata
import numpy as np
import torch
from eodata.grids import T1H
from eodata.normalization import ModalityNormalizer
from eodata.registry import get_data_reader_class
from eodata.sample import Mask, mask_name
from tensordict import TensorDict
from torch.utils.data import Dataset

from deep_lst.dataset.transforms import with_transform
from deep_lst.utils import features
from deep_lst.utils.io import ExperimentPaths, load_computed_stats, stats_spatial
from deep_lst.utils.spatial import anchor_grid, map_to_store_rows, store_grid

# Modality carrying the global time features (doy/hod); the finest / target grid.
_ANCHOR_MODALITY = "seviri"
# non-"data" data types -> the reader flag that enables reading them (data is always returned)
DATA_TYPES_ENABLE = {
    "metadata": "get_metadata",
    "data_daily": "get_data_daily",
    "data_static": "get_data_static",
    "data_static_categorical": "get_data_static_categorical",
}
# data type -> the reader's underlying zarr array attribute (its dtype fields give the full var order)
DATA_TYPES = {
    "data": "data_arr",
    "metadata": "metadata_arr",
    "data_daily": "data_daily_arr",
    "data_static": "data_static_arr",
    "data_static_categorical": "data_static_categorical_arr",
}

PATHS = ExperimentPaths()

# time_indices_*.npz are hours since the data start (2016-01-01); the new full stores are indexed
# on the 1970 epoch, so a time index must be shifted by this offset to address the store.
# FLAGGED for later: cleaner to regenerate the indices as absolute T1H (see migration notes).
STORE_EPOCH_OFFSET = T1H.date_to_index("2016-01-01")  # 403224 hours = 16801 days
PRELOAD_START_DAY = STORE_EPOCH_OFFSET // 24  # 2016-01-01
PRELOAD_END_DAY = T1H.date_to_index("2026-01-01") // 24  # exclusive


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


def _flatten_hours(values, valid):
    """(days, 24, C) -> (days * 24, C); re-inject NaN where invalid (float leaves only)."""
    n_vars = values.shape[-1]
    column = values.reshape(-1, n_vars).clone()
    if valid is not None and column.is_floating_point():
        column[~valid.reshape(-1, n_vars)] = float("nan")
    return column


def _hourly_column(leaf, mask):
    """data / metadata: (days, C, 24, 1, 1) -> (days * 24, C)."""
    values = leaf.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # (days, 24, C)
    valid = None if mask is None else mask.squeeze(-1).squeeze(-1).permute(0, 2, 1)
    return _flatten_hours(values, valid)


def _daily_column(leaf, mask):
    """data_daily: (days, C, 1, 1) -> (days * 24, C), each day repeated across its 24 hours."""
    values = leaf.squeeze(-1).squeeze(-1).unsqueeze(1).expand(-1, 24, -1)  # (days, 24, C)
    valid = None if mask is None else mask.squeeze(-1).squeeze(-1).unsqueeze(1).expand(-1, 24, -1)
    return _flatten_hours(values, valid)


def _static_column(leaf, mask):
    """data_static(_categorical): (days, C, 1, 1) constant over days -> (C,)."""
    column = leaf[0].squeeze(-1).squeeze(-1).clone()
    if mask is not None and column.is_floating_point():
        column[~mask[0].squeeze(-1).squeeze(-1)] = float("nan")
    return column


def _preload_pixel(reader, normalizer, pixel, select=None, day_range=None):
    """Read one pixel's day-axis and return every configured leaf as a RAM array.

    The full leaf is read and normalized (so the positional full-set statistics align), then
    the config's variables are selected by channel — dropping a variable is a config change,
    never a statistics recompute. Hourly (data, metadata) and daily (data_daily) leaves flatten
    to a continuous hour axis (position = day * 24 + hour, the T1H index); static leaves collapse
    to one per-pixel vector. Missing values are re-injected as NaN via each leaf's data mask; int
    categoricals keep their fill. day_range=(start_day, end_day) restricts the read to a store-day
    window (default: the full preload axis). Returns {key: hourly/daily -> (n_days * 24, C);
    static -> (C,)}.
    """
    hourly_keys = ("data", "metadata")
    daily_keys = ("data_daily",)
    static_keys = ("data_static", "data_static_categorical")

    start_day, end_day = day_range if day_range is not None else (PRELOAD_START_DAY, PRELOAD_END_DAY)
    sample = reader.get_data_multi([(int(pixel), day) for day in range(start_day, end_day)])
    if normalizer is not None:  # None => raw values (e.g. statistics computation)
        sample = normalizer.normalize(sample)
    if select is None:  # no selection: keep every loaded leaf, all channels
        select = {key: slice(None) for key in sample.variables()}
    columns = {}
    for key, channels in select.items():
        leaf = sample[key][:, channels]  # select the config variables (channel axis) post-normalize
        mask = sample.get(mask_name(key, Mask.DATA))
        if mask is not None:
            mask = mask[:, channels]
        if key in static_keys:
            columns[key] = _static_column(leaf, mask)
        elif key in daily_keys:
            columns[key] = _daily_column(leaf, mask)
        elif key in hourly_keys:
            # Dispatch on shape, not name: T1H stores carry an hour axis (days, C, 24, 1, 1),
            # daily stores (e.g. glofas_v4) name their leaf `data` too but omit it (days, C, 1, 1).
            if leaf.ndim == 5:
                columns[key] = _hourly_column(leaf, mask)
            else:
                columns[key] = _daily_column(leaf, mask)
        else:
            raise ValueError(f"Unclassified preload leaf {key!r}")
    return columns


def _preload_columns(reader, normalizer, pixel_ids, select=None, num_workers=1, day_range=None):
    """Preload every pixel's day-axis into RAM, one stacked entry per configured leaf.

    Returns {leaf_key: tensor}: hourly/daily leaves -> (n_pixels, n_days * 24, C); static leaves
    -> (n_pixels, C). select maps each leaf to the channel indices of its config variables. With
    num_workers > 1 the per-pixel reads run on a shared-reader thread pool; ThreadPoolExecutor.map
    preserves order so rows stay aligned with pixel_ids.
    """

    def load(pixel):
        return _preload_pixel(reader, normalizer, pixel, select, day_range)

    if num_workers and num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            per_pixel = list(pool.map(load, pixel_ids))
    else:
        per_pixel = [load(pixel) for pixel in pixel_ids]
    return {key: torch.stack([pixel[key] for pixel in per_pixel]) for key in per_pixel[0]}


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


class UnimodalDataset(Dataset):
    """Streaming raw single-modality dataset — used only for statistics computation.

    Reads a small set of statistics days for each pixel (no normalization, no full-axis preload,
    no windowing), re-injecting NaN via each leaf's mask so eodata's WelfordMetric ignores missing
    values. Each item is the per-day raw leaf block {leaf: (days, C, 24)} (or (days, C) for static),
    which the statistics script flattens for base stats and diffs along the 24-hour axis (within-day)
    for the prognostic increments.
    """

    def __init__(self, reader, rows, stat_days):
        self.reader = reader
        self.rows = np.asarray(rows, dtype=np.int64)
        self.stat_days = [int(day) for day in stat_days]
        print(f"Initialized statistics dataset: {len(self.rows)} pixels × {len(self.stat_days)} days.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = self.reader.get_data_multi([(int(self.rows[index]), day) for day in self.stat_days])
        leaves = {}
        for leaf in sample.variables():
            values = sample[leaf]  # (days, C, 24, 1, 1) hourly/daily, or (days, C, 1, 1) static
            mask = sample.get(mask_name(leaf, Mask.DATA))
            if mask is not None and values.is_floating_point():
                values = values.clone()
                values[~mask] = float("nan")  # re-inject NaN; the metric ignores it
            leaves[leaf] = values.squeeze(-1).squeeze(-1)  # (days, C, 24) or (days, C)
        return leaves

    @classmethod
    def from_config(cls, modality, store, spatial, anchor, temporal, workers=1, store_root=None):
        """Stream the MTG pixel selection (spatial) over the statistics timestep subset (temporal)."""
        anchor_ids = _load_anchor_ids(spatial)
        rows = np.unique(map_to_store_rows(anchor_grid(anchor), anchor_ids, store_grid(modality)))
        stat_hours = np.load(PATHS.temporal_indices_path(temporal))["time_indices"].astype(np.int64)
        stat_days = np.unique((stat_hours + STORE_EPOCH_OFFSET) // 24)  # store days; each read in full (24 h)

        # Enable the full continuous leaves (data + data_daily + data_static), raw. Categoricals and
        # metadata are left out — they get no statistics. Seviri-only flags are signature-filtered.
        reader_kwargs = {
            "average_hourly": False,
            "get_data_static": True,
            "get_data_daily": True,
            "get_metadata": False,
            "get_data_static_categorical": False,
        }
        accepted = inspect.signature(get_data_reader_class(modality)).parameters
        reader_kwargs = {key: value for key, value in reader_kwargs.items() if key in accepted}
        reader = eodata.get_data_reader(
            modality, store=str(PATHS.eozarr_store_path(store, store_root)), **reader_kwargs
        )
        return cls(reader, rows, stat_days)


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

        Every anchor cell (MTG for training, seviri pixels for eval) is nearest-neighbour mapped
        to each modality's native store row; coarse modalities are deduped so only unique cells
        are read. Columns depend only on (anchors, store), not the temporal split, so this runs
        once — a full-axis cache per split would not fit in RAM.
        """
        paths = paths or PATHS
        spatial_name = kwargs["spatial"]
        stats_tag = stats_spatial(kwargs)
        anchor_ids = _load_anchor_ids(spatial_name, paths)
        src_grid = anchor_grid(kwargs.get("anchor", "mtg"))
        num_workers = kwargs.get("preload_workers", 8)

        # Disk cache: the preload is deterministic given the anchors, stores, load specs and
        # statistics, so later runs (restarts, HP/architecture sweeps) skip the store reads.
        # Model-role changes reuse the cache; editing a `load:` section or regenerating a stats
        # zarr changes the fingerprint and triggers a rebuild.
        fingerprint_spec = {
            "anchors": hashlib.sha256(np.ascontiguousarray(anchor_ids).tobytes()).hexdigest(),
            "modalities": {
                modality: {
                    "store": kwargs[modality]["store"],
                    "load": kwargs[modality]["load"],
                    "stats_mtime": os.path.getmtime(paths.stats_zarr_path(modality, stats_tag) / "zarr.json"),
                    "axis": [PRELOAD_START_DAY, PRELOAD_END_DAY],
                }
                for modality in modality_ids
            },
        }
        for modality in modality_ids:
            store_root = kwargs[modality].get("store_root")
            if store_root is not None:
                fingerprint_spec["modalities"][modality]["store_root"] = str(Path(store_root).resolve())
            transform = kwargs[modality].get("transform")
            if transform is not None:  # only when set, so caches built before the key existed stay valid
                fingerprint_spec["modalities"][modality]["transform"] = transform
        fingerprint = hashlib.sha256(json.dumps(fingerprint_spec, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = paths.preload_cache_path(fingerprint, spatial_name)
        if cache_path.exists():
            print(f"Loading preload cache {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        columns, inverse, var_names, cell_lonlat = {}, {}, {}, {}
        for modality in modality_ids:
            native_grid = store_grid(modality)
            rows = map_to_store_rows(src_grid, anchor_ids, native_grid)  # (n_anchor,), may repeat
            unique_rows, inv = np.unique(rows, return_inverse=True)  # dedup coarse modalities

            # Enable exactly the leaves the config `load` names; read them FULL (no variable
            # subset) so the positional full-set statistics align, then select the config
            # variables after normalization (see _preload_pixel). Enable-flags for data types
            # this modality's reader doesn't accept (ERA5 has no metadata/data_daily) are
            # filtered out by the reader's __init__ signature.
            load = kwargs[modality]["load"]
            reader_kwargs = {"average_hourly": False}
            for data_type, flag in DATA_TYPES_ENABLE.items():
                reader_kwargs[flag] = data_type in load
            accepted = inspect.signature(get_data_reader_class(modality)).parameters
            reader_kwargs = {key: value for key, value in reader_kwargs.items() if key in accepted}
            reader = eodata.get_data_reader(
                modality,
                store=str(paths.eozarr_store_path(kwargs[modality]["store"], kwargs[modality].get("store_root"))),
                **reader_kwargs,
            )

            normalizer = ModalityNormalizer.from_spec(
                f"zscore:{paths.stats_zarr_path(modality, stats_tag)}", reader.schema
            )
            normalizer = with_transform(normalizer, kwargs[modality].get("transform"))
            # Channel indices of the config variables within each data type's full store variable order.
            select = {
                data_type: [list(getattr(reader, DATA_TYPES[data_type]).dtype.fields).index(var) for var in names]
                for data_type, names in load.items()
            }
            columns[modality] = _preload_columns(reader, normalizer, unique_rows, select, num_workers=num_workers)
            inverse[modality] = inv
            var_names[modality] = {leaf: list(names) for leaf, names in load.items()}
            cell_lonlat[modality] = np.asarray(native_grid.centers[unique_rows], dtype=np.float64)
        shared = {
            "columns": columns,
            "inverse": inverse,
            "var_names": var_names,
            "cell_lonlat": cell_lonlat,
            "anchor_ids": anchor_ids,
            "spatial": spatial_name,
        }
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
