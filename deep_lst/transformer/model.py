"""
Transformer model placeholder for deep_lst.

This module will contain the transformer implementation for land surface temperature estimation.
"""

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """Vision Transformer model placeholder."""

    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        raise NotImplementedError("ViT model is not yet implemented.")

    def name(self):
        return "vit"

    def forward(self, *inputs) -> torch.Tensor:
        raise NotImplementedError("ViT model is not yet implemented.")
