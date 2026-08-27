import torch.nn as nn

from deep_lst.models.static_embeddings import build_embedder
from deep_lst.models.transformer.encodings import PositionalEncoding
from deep_lst.models.transformer.tokenizer import Tokenizer


class Transformer(nn.Module):
    def __init__(self, n_prognostic, n_diagnostic, n_input, n_forcing_context, lookback, config, static_spec=None):
        super().__init__()

        self.n_prognostic = n_prognostic
        self.lookback = lookback
        self.target_size = n_prognostic + n_diagnostic

        # Same static block as the MLP, applied to the query token's forcing vector, so both
        # backbones see an identical feature vector (models/static_embeddings.py).
        self.static_embedder = build_embedder(n_input, config, static_spec)
        self.input_size = n_input if self.static_embedder is None else self.static_embedder.out_dim

        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.dim_feedforward = config["dim_feedforward"]

        self.dropout_p = config["dropout"]

        self.tokenizer = Tokenizer(n_forcing_context, n_prognostic, self.input_size, self.d_model)
        self.positional_encoding = PositionalEncoding(lookback + 1, self.d_model)

        # local template only: nn.TransformerEncoder deep-copies it n_layers times,
        # registering it as an attribute would add a third, unused parameter set (breaks DDP)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_p,
            activation=config.get("activation", "relu"),
            norm_first=config.get("norm_first", False),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.head = nn.Linear(self.d_model, n_prognostic + n_diagnostic)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(lookback + 1)
        self.register_buffer("causal_mask", causal_mask)

    def name(self):
        return "transformer"

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        N, L, H, W, C = state_window.shape

        if self.static_embedder is not None:
            step_X_dyn = self.static_embedder(step_X_dyn)

        # Tokenize the input
        tokens = self.tokenizer(state_window, X_dyn_window, step_X_dyn)
        tokens = tokens + self.positional_encoding(tokens)

        hidden = self.encoder(tokens, mask=self.causal_mask)

        # Take the last token (corresponding to the current step) for prediction
        last_token = hidden[:, -1]

        output = self.head(last_token)

        return output.reshape(N, H, W, -1)
