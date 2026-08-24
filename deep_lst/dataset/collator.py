from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from tensordict import TensorDict
from tensordict import stack as td_stack


@dataclass
class Collator:
    model_id: str
    packing_keys: dict | None = field(default=None)

    def __post_init__(self):
        dispatch: dict[str, Callable] = {
            "mlp": self.pack,
            "transformer": self.pack,
        }
        self._collate: Callable = dispatch[self.model_id]
        print(f"[Collator] model_id={self.model_id}, collate={self._collate.__name__}")

    def __call__(self, batch):
        return self._collate(batch)

    def _pack(self, td, keys):
        tensors = [td[key].float() for key in keys]
        return torch.cat(tensors, dim=-1)

    def pack(self, batch):
        td = td_stack(batch, dim=0)
        if self.packing_keys is None:
            return td

        batch_size = td.batch_size[0]
        return TensorDict(
            {
                "y_prog": self._pack(td, self.packing_keys["state_keys"]),
                "X_dyn": self._pack(td, self.packing_keys["input_keys"]),
                "X_dyn_ctx": self._pack(td, self.packing_keys["forcing_context_keys"]),
                "spatial_idx": td["spatial_idx"],
                "time_idx": td["time_idx"],
            },
            batch_size=[batch_size],
        )
