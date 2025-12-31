from deep_lst.dataset import UnimodalDataset, MultimodalDataset
from deep_lst.eo_statistics.time_filter import TemporalFilter
from wedata.data_reader.era5 import Era5Reader as Reader
from wedata.eo_zarr import EOZarrStore

from deep_lst.utils.misc import open_config

from pathlib import Path
import numpy as np
import code

CONFIG = open_config(experiment="test", path="experiments")

store_path = Path(CONFIG['store_path'] + CONFIG['modality']['era5']['store_name'])
indices_path = Path(CONFIG['assets_path'] + "era5/" + CONFIG['experiment'] + "/indices/indices.npz")
statistics_path = Path(CONFIG['assets_path'] + "era5/" + CONFIG['experiment'] + "/statistics/statistics.npz")

spatial_indices = np.load(indices_path)['mapping']
tf = TemporalFilter.from_range("2017-01-01 00:00", "2017-12-31 23:00")
temporal_indices = tf.filter_by_date(CONFIG['train_start'], CONFIG['train_end'])
temporal_indices = tf.stratified_sample(temporal_indices, frac=0.05, by="month_hour", seed=42)

store = EOZarrStore(store_path, mode='r+')

reader = Reader(eozarr_store=store)

reader.available_indices(indices=spatial_indices)
reader.set_normalizer(file = statistics_path)

dataset = UnimodalDataset(
        reader=reader,
        spatial_indices=spatial_indices,
        temporal_indices=temporal_indices
        )

code.interact(local=locals())