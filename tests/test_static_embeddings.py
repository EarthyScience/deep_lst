"""Tests for the learnable categorical-static embeddings (integer vs embedding paths)."""

import json

import pytest
import torch

from deep_lst.models.mlp.model import MultiModalMLP
from deep_lst.models.static_embeddings import StaticEmbedder, build_embedder
from deep_lst.models.transformer.model import Transformer
from deep_lst.preprocessing.static_vocab import embedding_dim

N_INPUT = 8
N_PROGNOSTIC = 1
N_DIAGNOSTIC = 0
N_FORCING_CONTEXT = 4
LOOKBACK = 6

# Two categoricals at channels 2 and 5, with non-contiguous codes including a negative fill.
SPEC = [
    {"name": "seviri/LWMASK", "channel": 2, "codes": [-1, 1], "dim": 3},
    {"name": "seviri/LANDCOV", "channel": 5, "codes": [-1, 2, 7, 16], "dim": 3},
]

MLP_CONFIG = {
    "hidden_sizes": [16, 8],
    "dropout": 0.0,
    "activation": "gelu",
    "static_encoding": "embedding",
}
TRANSFORMER_CONFIG = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "dim_feedforward": 32,
    "dropout": 0.0,
    "activation": "gelu",
    "static_encoding": "embedding",
}


def _batch(n=4, codes=(1, 7)):
    """A packed batch whose two categorical channels carry valid codes."""
    state = torch.randn(n, LOOKBACK, 1, 1, N_PROGNOSTIC)
    context = torch.randn(n, LOOKBACK, 1, 1, N_FORCING_CONTEXT)
    step = torch.randn(n, 1, 1, N_INPUT)
    step[..., 2] = codes[0]
    step[..., 5] = codes[1]
    return state, context, step


def test_embedder_output_width_and_channel_accounting():
    embedder = StaticEmbedder(N_INPUT, SPEC)
    # 8 raw channels - 2 categorical + (3 + 3) embedded
    assert embedder.out_dim == N_INPUT - 2 + 6
    out = embedder(_batch()[2])
    assert out.shape[-1] == embedder.out_dim
    assert torch.isfinite(out).all()


def test_mapping_is_deterministic():
    """Two builds from the same vocabulary produce identical lookup tables."""
    first, second = StaticEmbedder(N_INPUT, SPEC), StaticEmbedder(N_INPUT, SPEC)
    assert torch.equal(first.lookup_0, second.lookup_0)
    assert torch.equal(first.lookup_1, second.lookup_1)
    # ids are contiguous and ordered by the sorted raw codes
    assert first.lookup_1[torch.tensor([-1, 2, 7, 16]) - first.min_code_1].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("unseen", [99, -50, 3])
def test_unknown_code_maps_to_reserved_id(unseen):
    """A code absent from the training vocabulary must not crash or alias a real class."""
    embedder = StaticEmbedder(N_INPUT, SPEC)
    step = _batch()[2]
    step[..., 5] = unseen  # LANDCOV code the training pixels never saw

    reserved = embedder.embeddings[1].num_embeddings - 1
    assert (embedder._ids(step[..., 5], 1) == reserved).all()

    out = embedder(step)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(embedder.embeddings[1].weight.grad).all()
    # only the reserved row receives gradient
    assert embedder.embeddings[1].weight.grad[:reserved].abs().sum() == 0


def test_nan_code_maps_to_reserved_id():
    embedder = StaticEmbedder(N_INPUT, SPEC)
    step = _batch()[2]
    step[..., 2] = float("nan")
    assert torch.isfinite(embedder(step)).all()


@pytest.mark.parametrize("model_cls,config", [(MultiModalMLP, MLP_CONFIG), (Transformer, TRANSFORMER_CONFIG)])
def test_input_size_matches_first_layer_weight(model_cls, config):
    """The declared input width must equal the actual first-layer weight shape."""
    kwargs = dict(
        n_prognostic=N_PROGNOSTIC,
        n_diagnostic=N_DIAGNOSTIC,
        n_input=N_INPUT,
        n_forcing_context=N_FORCING_CONTEXT,
        lookback=LOOKBACK,
    )
    embedded = model_cls(config=config, static_spec=SPEC, **kwargs)
    legacy = model_cls(config=dict(config, static_encoding="integer"), static_spec=None, **kwargs)

    if model_cls is MultiModalMLP:
        assert embedded.pre[0][0].weight.shape[1] == embedded.input_size
        assert legacy.pre[0][0].weight.shape[1] == legacy.input_size
        # embeddings replace 2 raw channels with 6 embedded ones
        assert embedded.input_size == legacy.input_size + 4
    else:
        assert embedded.tokenizer.step_projection.weight.shape[1] == embedded.input_size
        assert legacy.tokenizer.step_projection.weight.shape[1] == N_INPUT
        assert embedded.input_size == N_INPUT + 4

    for model in (embedded, legacy):
        out = model(*_batch())
        assert out.shape == (4, 1, 1, N_PROGNOSTIC + N_DIAGNOSTIC)
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_cls,config", [(MultiModalMLP, MLP_CONFIG), (Transformer, TRANSFORMER_CONFIG)])
def test_checkpoint_roundtrip_preserves_embeddings_and_vocab(model_cls, config, tmp_path):
    kwargs = dict(
        n_prognostic=N_PROGNOSTIC,
        n_diagnostic=N_DIAGNOSTIC,
        n_input=N_INPUT,
        n_forcing_context=N_FORCING_CONTEXT,
        lookback=LOOKBACK,
        config=config,
        static_spec=SPEC,
    )
    model = model_cls(**kwargs)
    with torch.no_grad():  # make the weights distinctive so a silent re-init would show
        model.static_embedder.embeddings[0].weight.add_(1.234)

    path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), path)
    restored = model_cls(**kwargs)
    restored.load_state_dict(torch.load(path, weights_only=True))

    assert torch.equal(restored.static_embedder.embeddings[0].weight, model.static_embedder.embeddings[0].weight)
    # the vocab mapping travels in the checkpoint too, not just the weights
    assert torch.equal(restored.static_embedder.lookup_1, model.static_embedder.lookup_1)

    model.eval(), restored.eval()
    with torch.no_grad():
        batch = _batch()
        assert torch.equal(model(*batch), restored(*batch))


def test_integer_path_leaves_input_untouched():
    """static_encoding='integer' must reproduce today's behaviour: no embedder, raw channels."""
    assert build_embedder(N_INPUT, {"static_encoding": "integer"}, SPEC) is None
    model = MultiModalMLP(
        n_prognostic=N_PROGNOSTIC,
        n_diagnostic=N_DIAGNOSTIC,
        n_input=N_INPUT,
        n_forcing_context=N_FORCING_CONTEXT,
        lookback=LOOKBACK,
        config=dict(MLP_CONFIG, static_encoding="integer"),
    )
    assert model.static_embedder is None
    assert model.input_size == N_INPUT + 2 * LOOKBACK * N_PROGNOSTIC + LOOKBACK * N_FORCING_CONTEXT


def test_integer_path_is_bitwise_identical_to_the_legacy_forward():
    """One packed batch through both paths: integer mode must match the pre-embedding code
    exactly, so the v13 reference stays reproducible."""
    torch.manual_seed(0)
    model = MultiModalMLP(
        n_prognostic=N_PROGNOSTIC,
        n_diagnostic=N_DIAGNOSTIC,
        n_input=N_INPUT,
        n_forcing_context=N_FORCING_CONTEXT,
        lookback=LOOKBACK,
        config=dict(MLP_CONFIG, static_encoding="integer"),
    )
    model.eval()
    state, context, step = _batch()

    # verbatim legacy forward (pre-static-embeddings)
    state_flat = model._flatten_window(state)
    dyn_flat = model._flatten_window(context, with_mask=False)
    legacy_input = torch.cat([step, dyn_flat, state_flat], dim=-1)
    legacy_input[torch.isnan(legacy_input)] = 0.0
    with torch.no_grad():
        hidden = model.pre(legacy_input)
        if model.res_block is not None:
            hidden = hidden + model.res_block(hidden)
        expected = model.out_layer(model.post(hidden))
        actual = model(state, context, step)

    assert torch.equal(actual, expected)
    # the raw class codes reach the network untouched in integer mode
    assert torch.equal(legacy_input[..., 2], step[..., 2])
    assert torch.equal(legacy_input[..., 5], step[..., 5])


def test_embedding_dim_rule():
    assert embedding_dim(3) == 3
    assert embedding_dim(8) == 3
    assert embedding_dim(9) == 6
    assert embedding_dim(32) == 6
    assert embedding_dim(100) == 8


def test_vocab_asset_is_loadable_and_sorted():
    """The persisted asset is the model contract; codes must be sorted for reproducibility."""
    from deep_lst.preprocessing.static_vocab import vocab_path
    from deep_lst.utils.io import ExperimentPaths

    path = vocab_path(ExperimentPaths(), "mtg_new")
    if not path.exists():
        pytest.skip("static vocabulary asset not built in this working tree")
    stored = json.loads(path.read_text())["vocab"]
    for modality, variables in stored.items():
        for var, entry in variables.items():
            assert entry["codes"] == sorted(entry["codes"]), f"{modality}/{var} codes unsorted"
            assert entry["unknown_id"] == len(entry["codes"])
