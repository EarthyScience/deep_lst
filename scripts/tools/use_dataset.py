import code

import torch
from tensordict import stack as td_stack  # tensordict.stack
from tensordict.nn import TensorDictModule

from deep_lst.dataset.datamodule import DataModule
from deep_lst.dataset.dataset import MultimodalDataset, UnimodalDataset
from deep_lst.mlp.regression import MultiModalMLP
from deep_lst.utils.misc import open_config

CONFIG = open_config(experiment="test", path="configs")


def collate_multimodal_tensordict(batch):
    """
    batch: list of samples
    each sample is dict: {modality_id: TensorDict}
    returns:
    dict: {modality_id: TensorDict with batch dim prepended}

    if pack_tensors, returns a TensorDict with additional inputs key that contains a tensor per modality stacked along channels.
    """
    return td_stack(batch, dim=0)


def variable_keys(config, modality, var_type: str) -> list[tuple[str, str]]:
    keys = []
    for var in config[modality]["variables"]:
        ((varname, meta),) = var.items()
        if meta.get("type") == var_type:
            keys.append((modality, varname))
    return keys


uni_dataset = UnimodalDataset.from_modality(
    modality_id="era5",
    config=CONFIG["datareader"]["era5"],
    split="test",
)

uniloader = torch.utils.data.DataLoader(
    uni_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)

b1 = next(iter(uniloader))

multi_dataset = MultimodalDataset.from_modalities(
    config=CONFIG["datareader"],
    split="test",
)

multiloader = torch.utils.data.DataLoader(
    multi_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)


b2 = next(iter(multiloader))

code.interact(local=locals())

data_module = DataModule(config=CONFIG)

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
