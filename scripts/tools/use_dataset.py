import code

import torch
from tensordict import stack as td_stack  # tensordict.stack
from tensordict.nn import TensorDictModule

from deep_lst.dataset.data_module import LitDataModule
from deep_lst.dataset.dataset import MultimodalDataset, UnimodalDataset
from deep_lst.mlp.model import MultiModalMLP
from deep_lst.utils.config import open_config
from deep_lst.utils.io import ExperimentPaths

paths = ExperimentPaths()
CONFIG = open_config(paths.config_path("mlp_diag"))


def collate_multimodal_tensordict(batch):
    """
    batch: list of samples
    each sample is dict: {modality_id: TensorDict}
    returns:
    dict: {modality_id: TensorDict with batch dim prepended}

    if pack_tensors, returns a TensorDict with additional inputs key that contains a tensor per modality stacked along channels.
    """
    return td_stack(batch, dim=0)


train_dataset = UnimodalDataset.from_modality(modality_id="era5", **CONFIG["train_dataset"])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)

b1 = next(iter(train_loader))

multi_dataset = MultimodalDataset.from_modalities(modality_ids=["era5", "seviri"], **CONFIG["train_dataset"])

multiloader = torch.utils.data.DataLoader(
    multi_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)


b2 = next(iter(multiloader))

code.interact(local=locals())

data_module = LitDataModule(config=CONFIG)

in_size = len(
    CONFIG["datareader"]["era5"]["inputs"]
    + CONFIG["datareader"]["seviri"]["inputs"]
    + CONFIG["datareader"]["era5"]["prognostics"]
    + CONFIG["datareader"]["seviri"]["prognostics"]
)
out_size = len(
    CONFIG["datareader"]["era5"]["prognostics"]
    + CONFIG["datareader"]["seviri"]["prognostics"]
    + CONFIG["datareader"]["era5"]["diagnostics"]
    + CONFIG["datareader"]["seviri"]["diagnostics"]
)
model = MultiModalMLP(input_size=in_size, target_size=out_size, config=CONFIG)


td_model = TensorDictModule(
    module=model,
    in_keys=[("inputs", "era5"), ("inputs", "seviri")],
    out_keys=[("predictions", "LST_hat")],
)

code.interact(local=locals())
