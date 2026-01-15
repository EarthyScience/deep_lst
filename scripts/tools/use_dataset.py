from deep_lst.dataset.dataset import UnimodalDataset, MultimodalDataset
from deep_lst.utils.misc import open_config
from deep_lst.models.regression import MultiModalMLP

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
    """
    modalities = list(batch[0].keys())
    out = {}
    for m in modalities:
        tds = [sample[m] for sample in batch]
        out[m] = td_stack(tds, dim=0)  # batch dimension
    out = _pack_modalities(out, modalities)
    return out

def _pack_modalities(batch: dict, modalities) -> TensorDict:
        # batch["era5"] and batch["seviri"] are TensorDicts with same batch_size
        bs = batch[modalities[0]].batch_size
        device = batch[modalities[0]].device

        td = TensorDict({}, batch_size=bs, device=device)

        # Put each modality under a nested key namespace
        for modality in modalities:
            for key, values in batch[modality].items():
                td.set((modality, key), values)

        return td

def variable_keys(config, modality, var_type: str) -> list[tuple[str, str]]:
    keys = []
    for var in CONFIG['modality'][modality].get('variables', []):
        (varname, meta), = var.items()
        if meta.get("type") == var_type:
            keys.append((modality, varname))
    return keys

# uni_dataset = UnimodalDataset.from_modality(
#     modality_id="era5",
#     config=CONFIG,
#     split="train",
# )
#
# uniloader = torch.utils.data.DataLoader(
#     uni_dataset,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=torch.stack,
# )
#
# b1 = next(iter(uniloader))

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

input_era = variable_keys(CONFIG, modality='era5', var_type="input")
input_seviri = variable_keys(CONFIG, modality='seviri', var_type="input")
b2 = b2.set(("inputs", "era5"), torch.stack([b2.get(k) for k in input_era], dim=-1))
b2 = b2.set(("inputs", "seviri"), torch.stack([b2.get(k) for k in input_seviri], dim=-1))

model = MultiModalMLP(input_size = 3, target_size=1, config=CONFIG)

td_model = TensorDictModule(
        module=model,
        in_keys=[("inputs", "era5"), ("inputs", "seviri")],
        out_keys=[("predictions", "LST_hat")],
    )

code.interact(local=locals())
