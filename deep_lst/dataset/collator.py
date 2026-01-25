from collections.abc import Callable
from dataclasses import dataclass

import torch
from tensordict import stack as td_stack


@dataclass
class Collator:
    model_id: str

    def __post_init__(self):
        dispatch: dict[str, Callable] = {
            "mlp": self.stack,
            "cnn": self.stack_and_permute,
            "resnet": self.stack_and_permute,
            "vit": self.stack_and_permute,
        }
        try:
            self._collate: Callable = dispatch[self.model_id]
        except KeyError as e:
            raise ValueError(f"Unknown model_id={self.model_id!r}. Known: {sorted(dispatch)}") from e

        print(f"[Collator] model_id={self.model_id}, collate={self._collate.__name__}")

    def __call__(self, batch):
        return self._collate(batch)

    def stack(self, batch):
        """
        batch: list of samples
        each sample is dict: {modality_id: TensorDict}
        returns:
        dict: {modality_id: TensorDict with batch dim prepended}

        if pack_tensors, returns a TensorDict with additional inputs key that contains a tensor per modality stacked along channels.
        """
        return td_stack(batch, dim=0)

    def stack_and_permute(self, batch):
        """
        Converts:
        (B,T,H,W,C) -> (B,T,C,H,W)
        (B,H,W,C)   -> (B,C,H,W)

        Assumend spatial order is (H,W).

        batch: list of samples
        each sample is dict: {modality_id: TensorDict}
        returns:
        dict: {modality_id: TensorDict with batch dim prepended}
        """

        def _channels_first(x: torch.Tensor):
            if x.ndim == 5:
                x = x.permute(0, 1, 4, 2, 3).contiguous()
            if x.ndim == 4:
                x = x.permute(0, 3, 1, 2).contiguous()
            return x

        td = td_stack(batch, dim=0)

        return td.apply(_channels_first)
