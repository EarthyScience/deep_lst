import torch
from torch.nn import Embedding, Module


class PositionalEncoding(Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.embedding = Embedding(seq_len, d_model)

    def forward(self, tokens):
        N, L, D = tokens.shape
        positions = torch.arange(L, device=tokens.device)
        return self.embedding(positions)
