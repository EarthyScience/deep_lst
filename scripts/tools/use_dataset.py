from deep_lst.dataset import UnimodalDataset, MultimodalDataset
from deep_lst.utils.time_filter import get_temporal_indices
from wedata.eo_zarr import EOZarrStore
from deep_lst.utils.misc import open_config

from pathlib import Path
import numpy as np
import code
import torch
from tensordict import stack as td_stack  # tensordict.stack

CONFIG = open_config(experiment="test", path="experiments")

def collate_multimodal_tensordict(batch):
    """
    batch: list of samples
      each sample is dict: {modality_id: TensorDict}
    returns:
      dict: {modality_id: TensorDict with batch dim prepended}
    """
    modalities = batch[0].keys()
    out = {}
    for m in modalities:
        tds = [sample[m] for sample in batch]
        out[m] = td_stack(tds, dim=0)  # batch dimension
    return out


uni_dataset = UnimodalDataset.from_modality(
    modality_id="era5",
    config=CONFIG,
)

multi_dataset = MultimodalDataset.from_modalities(
    modality_ids=["era5", "seviri"],
    config=CONFIG,
)

uniloader = torch.utils.data.DataLoader(
    uni_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=torch.stack,
)

multiloader = torch.utils.data.DataLoader(
    multi_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)

b1 = next(iter(uniloader))
b2 = next(iter(multiloader))

code.interact(local=locals())
