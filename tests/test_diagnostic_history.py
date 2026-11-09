"""The diagnostic_history flag widens the state window to nP + nD channels in both backbones;
off, both backbones keep their previous input width."""

import pytest
import torch

from deep_lst.models.mlp.model import MultiModalMLP
from deep_lst.models.transformer.model import Transformer

N_INPUT = 8
N_PROGNOSTIC = 0
N_DIAGNOSTIC = 1
N_CONTEXT = 4
LOOKBACK = 6
MLP_CONFIG = {"hidden_sizes": [16, 16], "dropout": 0.0, "activation": "gelu", "static_encoding": "integer"}
TRANSFORMER_CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "static_encoding": "integer",
}


def _inputs(n_window, n=3):
    state = torch.randn(n, LOOKBACK, 1, 1, n_window)
    context = torch.randn(n, LOOKBACK, 1, 1, N_CONTEXT)
    step = torch.randn(n, 1, 1, N_INPUT)
    return state, context, step


@pytest.mark.parametrize("diagnostic_history, n_window", [(False, 0), (True, 1)])
def test_mlp_window_width(diagnostic_history, n_window):
    config = {**MLP_CONFIG, "diagnostic_history": diagnostic_history}
    model = MultiModalMLP(N_PROGNOSTIC, N_DIAGNOSTIC, N_INPUT, N_CONTEXT, LOOKBACK, config)

    assert model.input_size == N_INPUT + 2 * LOOKBACK * n_window + LOOKBACK * N_CONTEXT
    out = model(*_inputs(n_window))
    assert out.shape == (3, 1, 1, N_PROGNOSTIC + N_DIAGNOSTIC)


@pytest.mark.parametrize("diagnostic_history, n_window", [(False, 0), (True, 1)])
def test_transformer_window_width(diagnostic_history, n_window):
    config = {**TRANSFORMER_CONFIG, "diagnostic_history": diagnostic_history}
    model = Transformer(N_PROGNOSTIC, N_DIAGNOSTIC, N_INPUT, N_CONTEXT, LOOKBACK, config)

    assert model.tokenizer.window_projection.in_features == 2 * n_window + N_CONTEXT
    out = model(*_inputs(n_window))
    assert out.shape == (3, 1, 1, N_PROGNOSTIC + N_DIAGNOSTIC)


def test_a_diagnostic_copy_of_a_prognostic_adds_no_window_channel():
    """The joint head (LST prognostic and diagnostic) keeps the plain model's window width."""
    config = {**MLP_CONFIG, "diagnostic_history": True}
    model = MultiModalMLP(1, 1, N_INPUT, N_CONTEXT, LOOKBACK, config, n_diagnostic_history=0)

    assert model.input_size == N_INPUT + 2 * LOOKBACK * 1 + LOOKBACK * N_CONTEXT
