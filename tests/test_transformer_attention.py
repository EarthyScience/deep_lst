"""Tests for the transformer's attention option (model.attention: causal | full)."""

import pytest
import torch

from deep_lst.models.transformer.model import Transformer

N_INPUT = 8
N_PROGNOSTIC = 1
N_DIAGNOSTIC = 0
N_FORCING_CONTEXT = 4
LOOKBACK = 6

CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 2,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "activation": "gelu",
    "static_encoding": "integer",
}


def _model(attention):
    torch.manual_seed(0)
    config = {**CONFIG, "attention": attention}
    model = Transformer(N_PROGNOSTIC, N_DIAGNOSTIC, N_INPUT, N_FORCING_CONTEXT, LOOKBACK, config)
    return model.eval()


def _batch(n=3):
    state = torch.randn(n, LOOKBACK, 1, 1, N_PROGNOSTIC)
    context = torch.randn(n, LOOKBACK, 1, 1, N_FORCING_CONTEXT)
    step = torch.randn(n, 1, 1, N_INPUT)
    return state, context, step


def _hidden(model, state, context, step):
    """Encoder output (N, L + 1, d_model), computed the way forward() does."""
    tokens = model.tokenizer(state, context, step)
    tokens = tokens + model.positional_encoding(tokens)
    return model.encoder(tokens, mask=model.causal_mask)


@pytest.mark.parametrize("attention", ["causal", "full"])
def test_forward_shape(attention):
    model = _model(attention)
    out = model(*_batch())
    assert out.shape == (3, 1, 1, N_PROGNOSTIC + N_DIAGNOSTIC)
    assert torch.isfinite(out).all()


def test_default_is_causal():
    model = Transformer(N_PROGNOSTIC, N_DIAGNOSTIC, N_INPUT, N_FORCING_CONTEXT, LOOKBACK, CONFIG)
    assert model.causal_mask.shape == (LOOKBACK + 1, LOOKBACK + 1)
    assert "causal_mask" in model.state_dict()


def test_full_has_no_mask():
    model = _model("full")
    assert model.causal_mask is None
    assert "causal_mask" not in model.state_dict()


def test_unknown_attention_rejected():
    with pytest.raises(ValueError):
        _model("bidirectional")


@pytest.mark.parametrize("attention,history_sees_query", [("causal", False), ("full", True)])
def test_history_tokens_see_the_query_token_only_under_full_attention(attention, history_sees_query):
    """Perturbing the query (last) token changes the first history token's hidden state
    under full attention and leaves it untouched under the causal mask."""
    model = _model(attention)
    state, context, step = _batch()
    with torch.no_grad():
        hidden = _hidden(model, state, context, step)
        hidden_perturbed = _hidden(model, state, context, step + 1.0)
    first_token_changed = not torch.allclose(hidden[:, 0], hidden_perturbed[:, 0])
    assert first_token_changed == history_sees_query
