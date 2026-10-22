"""Learnable embeddings for the integer-coded categorical static inputs.

Backbone-agnostic on purpose: this runs on the packed input vector before the backbone's
first layer, so every model (MultiModalMLP, Transformer, ...) sees the same feature vector
and architecture comparisons stay clean.

The categorical channels are replaced, not augmented: the n raw code channels are dropped
from the continuous block and sum(dim) embedded channels are appended.
"""

import torch
import torch.nn as nn


class StaticEmbedder(nn.Module):
    """Raw class codes in the packed input -> per-variable embeddings.

    specs: [{"name": str, "channel": int, "codes": [raw code, ...], "dim": int}, ...]
    where `channel` is the variable's position in the packed input vector and `codes` is the
    training vocabulary (see preprocessing/static_vocab.py). Any code outside `codes` — a class
    the training pixels never saw — maps to the reserved trailing id.
    """

    def __init__(self, n_input: int, specs: list[dict]):
        super().__init__()

        self.n_input = n_input
        self.names = [spec["name"] for spec in specs]
        self.dims = [spec["dim"] for spec in specs]

        categorical_channels = [spec["channel"] for spec in specs]
        continuous = [c for c in range(n_input) if c not in set(categorical_channels)]
        self.register_buffer("continuous_channels", torch.tensor(continuous, dtype=torch.long))
        self.register_buffer("categorical_channels", torch.tensor(categorical_channels, dtype=torch.long))

        self.embeddings = nn.ModuleList()
        for index, spec in enumerate(specs):
            codes = [int(code) for code in spec["codes"]]
            unknown_id = len(codes)
            # Dense raw-code -> contiguous-id table, offset by the smallest code so negative
            # fills (LWMASK/LANDCOV use -1) index normally. Registered as a buffer: it travels
            # with the model to the GPU and into the checkpoint, so a loaded model cannot
            # silently use a different mapping than the one it was trained with.
            lookup = torch.full((max(codes) - min(codes) + 1,), unknown_id, dtype=torch.long)
            for contiguous_id, code in enumerate(codes):
                lookup[code - min(codes)] = contiguous_id
            self.register_buffer(f"lookup_{index}", lookup)
            self.register_buffer(f"min_code_{index}", torch.tensor(min(codes), dtype=torch.long))
            self.embeddings.append(nn.Embedding(unknown_id + 1, spec["dim"]))

        self.out_dim = len(continuous) + sum(self.dims)

    def _ids(self, values: torch.Tensor, index: int) -> torch.Tensor:
        """Raw codes -> contiguous ids; unseen codes and NaN fills go to the reserved id."""
        lookup = getattr(self, f"lookup_{index}")
        min_code = getattr(self, f"min_code_{index}")
        unknown_id = int(self.embeddings[index].num_embeddings) - 1

        # NaN would make .long() undefined; send it out of range so it lands on unknown.
        codes = torch.nan_to_num(values, nan=float(int(min_code) - 1)).round().long()
        offset = codes - min_code
        inside = (offset >= 0) & (offset < lookup.numel())
        ids = lookup[offset.clamp(0, lookup.numel() - 1)]
        return torch.where(inside, ids, torch.full_like(ids, unknown_id))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(..., n_input) -> (..., out_dim): continuous channels then embedded categoricals."""
        parts = [x.index_select(-1, self.continuous_channels)]
        for index, channel in enumerate(self.categorical_channels.tolist()):
            ids = self._ids(x[..., channel], index)
            parts.append(self.embeddings[index](ids))
        return torch.cat(parts, dim=-1)

    def extra_repr(self) -> str:
        pairs = ", ".join(f"{name}:{dim}" for name, dim in zip(self.names, self.dims))
        return f"n_input={self.n_input}, out_dim={self.out_dim}, [{pairs}]"


def build_embedder(n_input: int, config: dict, static_spec: list[dict] | None) -> StaticEmbedder | None:
    """StaticEmbedder for model.static_encoding == 'embedding', else None (legacy raw codes).

    The legacy 'integer' path is kept so the v13 reference runs stay reproducible.
    """
    encoding = config.get("static_encoding", "embedding")
    if encoding == "integer":
        return None
    if encoding != "embedding":
        raise ValueError(f"model.static_encoding must be 'integer' or 'embedding', got {encoding!r}")
    if not static_spec:
        raise ValueError(
            "model.static_encoding='embedding' needs a static_spec; build the vocabulary with "
            "`python -m deep_lst.dataset.static_vocab` and pass static_spec into the model."
        )
    return StaticEmbedder(n_input, static_spec)
