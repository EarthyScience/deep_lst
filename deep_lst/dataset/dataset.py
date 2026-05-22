import eodata
import numpy as np
from eodata.eo_zarr import EOZarrStore
from eodata.grids import T1H
from eodata.util.misc import assemble_statistics_from_npz
from tensordict import TensorDict
from torch.utils.data import Dataset

from deep_lst.utils.io import ExperimentPaths

PATHS = ExperimentPaths()


def load_samples(
    path: str,
    start_time: str | None = None,
    end_time: str | None = None,
    temporal_indices_path: str | list | None = None,
    horizon: int = 0,
    lookback: int = 0,
) -> np.ndarray:
    data = np.load(path)
    space_idx = data["mapping"]
    order = np.argsort(data["src_indices"], kind="stable")
    space_idx = space_idx[order]

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

    space, time = np.meshgrid(space_idx, time_idx, indexing="ij")
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
        indices = load_samples(
            PATHS.spatial_indices_path(kwargs["spatial"]),
            start_time=kwargs.get("start_time"),
            end_time=kwargs.get("end_time"),
            temporal_indices_path=temporal_paths,
            horizon=kwargs[modality_id].get("horizon", 0),
            lookback=kwargs[modality_id].get("lookback", 0),
        )
        statistics = assemble_statistics_from_npz(PATHS.statistics_path(modality_id, kwargs[modality_id]["statistics"]))

        if kwargs[modality_id].get("preload", False):
            store_spatial = kwargs.get("sampled_spatial", kwargs["spatial"])
            store = EOZarrStore(PATHS.sampled_store_path(modality_id, store_spatial), mode="r+")
        else:
            store = EOZarrStore(PATHS.store_path(modality_id), mode="r+")

        # Instantiate reader
        reader = eodata.get_data_reader(modality_id, eozarr_store=store, **kwargs[modality_id])
        reader.set_normalizer(statistics=statistics)

        if kwargs[modality_id].get("preload", False):
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
        indices = {
            mod: load_samples(
                PATHS.spatial_indices_path(mod, kwargs["spatial"]),
                start_time=kwargs.get("start_time"),
                end_time=kwargs.get("end_time"),
                temporal_indices_path=temporal_paths,
                horizon=kwargs[mod].get("horizon", 0),
                lookback=kwargs[mod].get("lookback", 0),
            )
            for mod in modality_ids
        }

        readers = {}
        for mod in modality_ids:
            if kwargs[mod].get("preload", False):
                store_spatial = kwargs.get("sampled_spatial", kwargs["spatial"])
                store = EOZarrStore(PATHS.sampled_store_path(mod, store_spatial), mode="r+")
            else:
                store = EOZarrStore(PATHS.store_path(mod), mode="r+")
            reader = eodata.get_data_reader(mod, eozarr_store=store, **kwargs[mod])
            reader.set_normalizer(
                statistics=assemble_statistics_from_npz(PATHS.statistics_path(mod, kwargs[mod]["statistics"]))
            )
            if kwargs[mod].get("preload", False):
                reader.preload()
            readers[mod] = reader

        dataset = cls(
            reader=readers,
            indices=indices,
            **kwargs,
        )

        return dataset
