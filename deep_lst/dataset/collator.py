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

    def _pack(self, td, keys):
        """Concatenate the given (modality, var) channels along the last axis, prog-first
        order preserved. Cast to float so categoricals (int8) sit beside continuous vars."""
        tensors = [td[key].float() for key in keys]
        return torch.cat(tensors, dim=-1)

    def pack(self, batch):
        """Pack the per-(modality,var) sample dicts into the dense tensors the packed model
        and loss consume. Every tensor is (B, L+H+1, H, W, C); NaNs are preserved (the model
        imputes + masks). Channel order is fixed by packing_keys (see utils.config.get_packing_keys).

          y_prog     prognostics then diagnostics (the full state)
          X_dyn      all `inputs` (step forcings, statics/categoricals included)
          X_dyn_ctx  forcing-context vars (the lookback-window forcings)
        """
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
