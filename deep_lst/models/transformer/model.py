import torch.nn as nn

from deep_lst.models.static_embeddings import build_embedder
from deep_lst.models.transformer.encodings import PositionalEncoding
from deep_lst.models.transformer.tokenizer import Tokenizer


class Transformer(nn.Module):
    def __init__(
        self,
        n_prognostic,
        n_diagnostic,
        n_input,
        n_forcing_context,
        lookback,
        config,
        static_spec=None,
        n_diagnostic_history=None,
    ):
        super().__init__()

        self.n_prognostic = n_prognostic
        self.lookback = lookback
        self.target_size = n_prognostic + n_diagnostic
        # With diagnostic_history (default) the window also carries the observed past of the
        # diagnostics that are not prognostics as well (n_diagnostic_history of them), hidden from
        # t0+1 on by the loss and the inference loop.
        self.diagnostic_history = config.get("diagnostic_history", True)
        n_history = n_diagnostic if n_diagnostic_history is None else n_diagnostic_history
        n_window = n_prognostic + n_history if self.diagnostic_history else n_prognostic

        # Same static block as the MLP, applied to the query token's forcing vector, so both
        # backbones see an identical feature vector (models/static_embeddings.py).
        self.static_embedder = build_embedder(n_input, config, static_spec)
        self.input_size = n_input if self.static_embedder is None else self.static_embedder.out_dim

        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_layers = config["n_layers"]
        self.dim_feedforward = config["dim_feedforward"]

        self.dropout_p = config["dropout"]

        self.tokenizer = Tokenizer(n_forcing_context, n_window, self.input_size, self.d_model)
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

        # causal: token t attends to tokens <= t. The query token is last and sees the whole
        # window either way, so the mask only restricts how the history tokens build their
        # own representations in layers 1..n_layers-1. full: bidirectional over the window
        # (a None buffer is passed as mask=None below).
        attention = config.get("attention", "causal")
        if attention == "causal":
            causal_mask = nn.Transformer.generate_square_subsequent_mask(lookback + 1)
        elif attention == "full":
            causal_mask = None
        else:
            raise ValueError(f"model.attention must be 'causal' or 'full', got {attention!r}")
        self.register_buffer("causal_mask", causal_mask)

    def name(self):
        return "transformer"

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        N, L, H, W, C = state_window.shape

        if self.static_embedder is not None:
            step_X_dyn = self.static_embedder(step_X_dyn)

        tokens = self.tokenizer(state_window, X_dyn_window, step_X_dyn)
        tokens = tokens + self.positional_encoding(tokens)

        hidden = self.encoder(tokens, mask=self.causal_mask)

        # Take the last token (corresponding to the current step) for prediction
        last_token = hidden[:, -1]

        output = self.head(last_token)

        return output.reshape(N, H, W, -1)
