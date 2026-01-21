from deep_lst.dataset.dataset import UnimodalDataset, MultimodalDataset
from deep_lst.utils.misc import open_config
from deep_lst.models.regression import MultiModalMLP
from deep_lst.dataset.datamodule import DataModule

import numpy as np
import code
import torch
from tensordict import stack as td_stack  # tensordict.stack
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

CONFIG = open_config(experiment="test", path="experiments")


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
    for var in config[modality]['variables']:
        (varname, meta), = var.items()
        if meta.get("type") == var_type:
            keys.append((modality, varname))
    return keys

uni_dataset = UnimodalDataset.from_modality(
     modality_id="era5",
     config=CONFIG,
     split="train",
 )

uniloader = torch.utils.data.DataLoader(
     uni_dataset,
     batch_size=2,
     shuffle=True,
     collate_fn=collate_multimodal_tensordict,
)

b1 = next(iter(uniloader))

multi_dataset = MultimodalDataset.from_modalities(
    modality_ids=["era5", "seviri"],
    config=CONFIG,
    split="train",
)

multiloader = torch.utils.data.DataLoader(
    multi_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_multimodal_tensordict,
)


b2 = next(iter(multiloader))


data_module = DataModule(config=CONFIG, modality_ids=["era5", "seviri"])

in_size = len(CONFIG['era5']['inputs'] + CONFIG['seviri']['inputs'] + CONFIG['era5']['prognostics'] + CONFIG['seviri']['prognostics'])
out_size = len(CONFIG['era5']['prognostics'] + CONFIG['seviri']['prognostics'] + CONFIG['era5']['diagnostics'] + CONFIG['seviri']['diagnostics'])
model = MultiModalMLP(input_size = in_size , target_size=out_size, config=CONFIG, modality_ids=["era5", "seviri"])


td_model = TensorDictModule(
        module=model,
        in_keys=[("inputs", "era5"), ("inputs", "seviri")],
        out_keys=[("predictions", "LST_hat")],
    )

code.interact(local=locals())