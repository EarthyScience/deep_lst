import eodata
import numpy as np
import torch
import zarr
from eodata.eo_zarr import EOZarrStore
from eodata.grids import T1H, build_era5_grid, build_seviri_grid
from eodata.grids.majortom import MajorTomLandGrid
from eodata.grids.spatial import IndexedSubsetGrid
from eodata.util.misc import assemble_statistics_from_npz
from tensordict import TensorDict
from torch.utils.data import Dataset

from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()


def _spatial_idx_from_npz(path) -> np.ndarray:
    # Non-preload fallback: base zarrs have no spatial_idx array, so derive it from the npz.
    data = np.load(path)
    order = np.argsort(data["src_indices"], kind="stable")
    return data["mapping"][order]


def _parent_grid_for(index_space: str, modality: str):
    if index_space.startswith("major_tom_global"):
        # Placeholder: model never addresses spatial_idx for MTG stores, parent grid is cosmetic.
        return MajorTomLandGrid(5)
    if index_space.startswith("equiangular"):
        if modality == "seviri":
            return build_seviri_grid()
        if modality == "era5":
            return build_era5_grid()
        raise ValueError(f"No equiangular grid builder for modality {modality!r}")
    raise ValueError(f"Unknown index_space: {index_space!r}")


def load_samples(
    spatial_idx: np.ndarray,
    start_time: str | None = None,
    end_time: str | None = None,
    temporal_indices_path: str | list | None = None,
    horizon: int = 0,
    lookback: int = 0,
) -> np.ndarray:
    if temporal_indices_path is not None:
        time_idx = np.load(temporal_indices_path)["time_indices"].astype(np.int64)

        if horizon > 0:
            time_idx = time_idx[time_idx <= time_idx.max() - horizon]
        if lookback > 0:
            time_idx = time_idx[time_idx >= lookback]
    else:
        time_idx = np.arange(
            start=T1H.date_to_index(start_time) + lookback,
            stop=T1H.date_to_index(end_time) - horizon,
        )

    # Time-major: all pixels of init_time 0, then init_time 1, ... This aligns batches with
    # whole-init_time writes to forecast.zarr (chunked one init_time per chunk). Read-side
    # order is irrelevant: inference preloads to RAM, training shuffles, statistics aggregate.
    time, space = np.meshgrid(time_idx, spatial_idx, indexing="ij")
    indices = np.stack([space.ravel(), time.ravel()], axis=1)  # (N_time * N_spatial, 2)
    return indices


class UnimodalDataset(Dataset):
    def __init__(self, reader, indices):
        """
        zarr_array: xarray.DataArray or zarr.Array
        indices: 3d pandas dataframe or array
        source: pandas data frame, read from source parquet
        """
        self.indices = indices
        self.reader = reader
        print(f"Initialized Unimodal dataset with {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        p_index, t_index = self.indices[idx]
        sample = self.reader.get_data_single(spatial_idx=p_index, time_idx=t_index)
        sample["spatial_idx"] = torch.tensor(p_index, dtype=torch.long)
        sample["time_idx"] = torch.tensor(t_index, dtype=torch.long)
        return sample

    @property
    def time_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 1])

    @property
    def spatial_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 0])

    @classmethod
    def from_modality(
        cls,
        modality_id: str,
        **kwargs: dict,
    ):
        temporal_split = kwargs["temporal"]
        if isinstance(temporal_split, list):
            temporal_paths = [PATHS.temporal_indices_path(p) for p in temporal_split]
        elif temporal_split is not None:
            temporal_paths = PATHS.temporal_indices_path(temporal_split)
        else:
            temporal_paths = None

        preload = kwargs[modality_id].get("preload", False)
        if preload:
            store_path = PATHS.sampled_store_path(modality_id, kwargs["spatial"])
            root = zarr.open_group(store_path, mode="r")
            spatial_idx = root["spatial_idx"][:]
            sampled_grid = IndexedSubsetGrid(
                _parent_grid_for(root.attrs["index_space"], modality_id),
                spatial_idx,
                name=root.attrs["spatial_grid"],
            )
            store = EOZarrStore(store_path, spatial_grid=sampled_grid, time_grid=T1H, mode="r+")
        else:
            spatial_idx = _spatial_idx_from_npz(PATHS.spatial_indices_path(modality_id, kwargs["spatial"]))
            store = EOZarrStore(PATHS.store_path(modality_id), mode="r+")

        indices = load_samples(
            spatial_idx,
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            temporal_indices_path=temporal_paths,
            horizon=kwargs[modality_id].get("horizon", 0),
            lookback=kwargs[modality_id].get("lookback", 0),
        )
        statistics = assemble_statistics_from_npz(PATHS.statistics_path(modality_id, kwargs[modality_id]["statistics"]))

        # Instantiate reader
        reader = eodata.get_data_reader(modality_id, store=store, **kwargs[modality_id])
        reader.set_normalizer(statistics=statistics)

        if preload:
            reader.preload()

        return cls(reader=reader, indices=indices)


class MultimodalDataset(Dataset):
    def __init__(
        self,
        reader: dict,
        indices: np.ndarray,
        spatial_lookups: dict[str, np.ndarray],
        n_spatial: int,
    ):
        self.readers = reader
        self.indices = indices  # (N, 2): (seviri_pixel, time), time-major
        self.spatial_lookups = spatial_lookups  # modality → array of length n_seviri
        self.n_spatial = n_spatial
        self.samples = len(self.indices)
        print(
            f"Initialized MultimodalDataset with {self.samples} samples "
            f"({n_spatial} pixels × {self.samples // n_spatial} times, time-major)."
        )

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        seviri_pos = idx % self.n_spatial
        seviri_id, t_idx = self.indices[idx]

        data = TensorDict({}, batch_size=[])
        for modality, reader in self.readers.items():
            mod_spatial_id = int(self.spatial_lookups[modality][seviri_pos])
            data[modality] = reader.get_data_single(spatial_idx=mod_spatial_id, time_idx=int(t_idx))

        data["spatial_idx"] = torch.tensor(seviri_id, dtype=torch.long)
        data["time_idx"] = torch.tensor(t_idx, dtype=torch.long)
        return data

    @property
    def time_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 1])

    @property
    def spatial_idx(self) -> np.ndarray:
        return np.unique(self.indices[:, 0])

    @classmethod
    def from_modalities(cls, modality_ids: list[str], **kwargs: dict):
        # SEVIRI is hardcoded as the anchor (fine-grid source of truth). Formalise this
        # "anchor modality" concept when other multimodal combinations show up.
        if "seviri" not in modality_ids:
            raise ValueError("MultimodalDataset.from_modalities requires 'seviri' as the anchor modality.")

        temporal_split = kwargs["temporal"]
        if isinstance(temporal_split, list):
            temporal_paths = [PATHS.temporal_indices_path(p) for p in temporal_split]
        elif temporal_split is not None:
            temporal_paths = PATHS.temporal_indices_path(temporal_split)
        else:
            temporal_paths = None

        # Anchor on SEVIRI: every (seviri_pixel, time) sample maps to the per-modality
        # spatial id via that modality's npz (src_indices → mapping).
        seviri_kwargs = kwargs["seviri"]
        if seviri_kwargs.get("preload", False):
            seviri_root = zarr.open_group(PATHS.sampled_store_path("seviri", kwargs["spatial"]), mode="r")
            seviri_spatial_idx = seviri_root["spatial_idx"][:]
        else:
            seviri_spatial_idx = _spatial_idx_from_npz(PATHS.spatial_indices_path("seviri", kwargs["spatial"]))

        # Clip by the strictest horizon/lookback across modalities so every reader can serve its window.
        shared_horizon = max(kwargs[m].get("horizon", 0) for m in modality_ids)
        shared_lookback = max(kwargs[m].get("lookback", 0) for m in modality_ids)
        indices = load_samples(
            seviri_spatial_idx,
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            temporal_indices_path=temporal_paths,
            horizon=shared_horizon,
            lookback=shared_lookback,
        )
        n_spatial = len(seviri_spatial_idx)

        spatial_lookups: dict[str, np.ndarray] = {}
        readers: dict[str, object] = {}
        for mod in modality_ids:
            # Per-modality SEVIRI-position → modality-id lookup, from the npz mapping.
            # MTG: src_indices == mapping → identity. Eval region: fine SEVIRI → coarse ERA5.
            npz = np.load(PATHS.spatial_indices_path(mod, kwargs["spatial"]))
            s_to_m = dict(zip(npz["src_indices"].astype(np.int64), npz["mapping"].astype(np.int64)))
            spatial_lookups[mod] = np.array([s_to_m[int(s)] for s in seviri_spatial_idx], dtype=np.int64)

            preload = kwargs[mod].get("preload", False)
            if preload:
                store_path = PATHS.sampled_store_path(mod, kwargs["spatial"])
                root = zarr.open_group(store_path, mode="r")
                mod_spatial_idx = root["spatial_idx"][:]
                sampled_grid = IndexedSubsetGrid(
                    _parent_grid_for(root.attrs["index_space"], mod),
                    mod_spatial_idx,
                    name=root.attrs["spatial_grid"],
                )
                store = EOZarrStore(store_path, spatial_grid=sampled_grid, time_grid=T1H, mode="r+")
            else:
                store = EOZarrStore(PATHS.store_path(mod), mode="r+")

            reader = eodata.get_data_reader(mod, store=store, **kwargs[mod])
            reader.set_normalizer(
                statistics=assemble_statistics_from_npz(PATHS.statistics_path(mod, kwargs[mod]["statistics"]))
            )
            if preload:
                reader.preload()
            readers[mod] = reader

        return cls(reader=readers, indices=indices, spatial_lookups=spatial_lookups, n_spatial=n_spatial)
