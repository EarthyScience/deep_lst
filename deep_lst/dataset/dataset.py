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

    space, time = np.meshgrid(spatial_idx, time_idx, indexing="ij")
    indices = np.stack([space.ravel(), time.ravel()], axis=1)  # (N_spatial * N_time, 2)
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
    def __init__(self, reader: dict[str], indices: dict[str], **kwargs):
        self.readers = reader
        self.indices = indices
        # self.kwargs = kwargs
        self.samples = len(self.indices[next(iter(self.readers))])
        print(f"Initialized MultimodalDataset with {self.samples} samples.")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        data = TensorDict({}, batch_size=[])

        for modality, reader in self.readers.items():
            p_index, t_index = self.indices[modality][idx]
            data[modality] = reader.get_data_single(spatial_idx=p_index, time_idx=t_index)

        ref_mod = "seviri"
        p_index, t_index = self.indices[ref_mod][idx]

        data["spatial_idx"] = torch.tensor(p_index, dtype=torch.long)
        data["time_idx"] = torch.tensor(t_index, dtype=torch.long)

        return data

    @classmethod
    def from_modalities(cls, modality_ids: list[str], **kwargs: dict):
        temporal_split = kwargs["temporal"]
        if isinstance(temporal_split, list):
            temporal_paths = [PATHS.temporal_indices_path(p) for p in temporal_split]
        elif temporal_split is not None:
            temporal_paths = PATHS.temporal_indices_path(temporal_split)
        else:
            temporal_paths = None

        indices = {}
        readers = {}
        for mod in modality_ids:
            preload = kwargs[mod].get("preload", False)
            if preload:
                store_path = PATHS.sampled_store_path(mod, kwargs["spatial"])
                root = zarr.open_group(store_path, mode="r")
                spatial_idx = root["spatial_idx"][:]
                sampled_grid = IndexedSubsetGrid(
                    _parent_grid_for(root.attrs["index_space"], mod),
                    spatial_idx,
                    name=root.attrs["spatial_grid"],
                )
                store = EOZarrStore(store_path, spatial_grid=sampled_grid, time_grid=T1H, mode="r+")
            else:
                spatial_idx = _spatial_idx_from_npz(PATHS.spatial_indices_path(mod, kwargs["spatial"]))
                store = EOZarrStore(PATHS.store_path(mod), mode="r+")

            indices[mod] = load_samples(
                spatial_idx,
                start_time=kwargs.get("start_time"),
                end_time=kwargs.get("end_time"),
                temporal_indices_path=temporal_paths,
                horizon=kwargs[mod].get("horizon", 0),
                lookback=kwargs[mod].get("lookback", 0),
            )

            reader = eodata.get_data_reader(mod, store=store, **kwargs[mod])
            reader.set_normalizer(
                statistics=assemble_statistics_from_npz(PATHS.statistics_path(mod, kwargs[mod]["statistics"]))
            )
            if preload:
                reader.preload()
            readers[mod] = reader

        dataset = cls(
            reader=readers,
            indices=indices,
            **kwargs,
        )

        return dataset
