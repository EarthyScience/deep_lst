"""Tests for deep_lst/losses/rollout.py

Hermetic: a stub model that returns one learnable value per output channel.

Covered:
  RolloutLoss with zero prognostics (diagnostic-only run, v18 no. 4)
    - the loss is finite and equals the teacher-forced diagnostic term
    - the rollout term is zero (nothing is carried between steps)
    - gradients reach the model
  RolloutLoss with one prognostic
    - the state and rollout terms stay active (guard for the nP > 0 path)
"""

import torch
import torch.nn.functional as F

from deep_lst.losses import RolloutLoss

B = 3
LOOKBACK = 4
HORIZON = 2
T = LOOKBACK + HORIZON + 1
N_FORCING = 2
N_CONTEXT = 1


class _ConstantModel(torch.nn.Module):
    """Returns one learnable value per output channel, broadcast over the batch."""

    def __init__(self, n_out):
        super().__init__()
        self.values = torch.nn.Parameter(torch.linspace(0.1, 0.5, n_out))

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        N, H, W, _ = step_X_dyn.shape
        return self.values.expand(N, H, W, -1)


class _StubModule:
    """The two LitModelModule attributes RolloutLoss reads, plus a log() that records values."""

    def __init__(self, model):
        self.model = model
        self.config = {
            "train_dataset": {"seviri": {"lookback": LOOKBACK, "horizon": HORIZON}},
            "model": {"loss_fn": "smooth_l1"},
        }
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


def _batch(n_states, seed=0):
    generator = torch.Generator().manual_seed(seed)
    y_prog = torch.randn(B, T, 1, 1, n_states, generator=generator)
    y_prog[0, LOOKBACK + 1, 0, 0, :] = float("nan")  # one masked target
    X_dyn = torch.randn(B, T, 1, 1, N_FORCING, generator=generator)
    X_dyn_ctx = torch.randn(B, T, 1, 1, N_CONTEXT, generator=generator)
    return {"y_prog": y_prog, "X_dyn": X_dyn, "X_dyn_ctx": X_dyn_ctx}


def _loss(prognostic_keys):
    return RolloutLoss(
        prognostic_keys=prognostic_keys,
        detach_rollout=False,
        teacher_forcing_loss=True,
        incremental_loss=False,
        rollout_loss=True,
    )


def test_zero_prognostics_gives_the_diagnostic_term_only():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)
    batch = _batch(n_states=1)

    total = _loss(prognostic_keys=[])(module, batch, stage="train")

    targets = batch["y_prog"][:, LOOKBACK + 1 :, ..., 0]
    valid = torch.isfinite(targets)
    expected = F.smooth_l1_loss(model.values[0].expand_as(targets)[valid], targets[valid])
    assert torch.isfinite(total)
    assert torch.isclose(total, expected)
    assert module.logged["train_loss_rollout"] == 0.0


def test_zero_prognostics_backpropagates():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)

    total = _loss(prognostic_keys=[])(module, _batch(n_states=1), stage="train")
    total.backward()

    assert torch.isfinite(model.values.grad).all()


def test_one_prognostic_keeps_state_and_rollout_terms():
    model = _ConstantModel(n_out=1)
    module = _StubModule(model)

    total = _loss(prognostic_keys=[("seviri", "LST")])(module, _batch(n_states=1), stage="train")

    assert torch.isfinite(total)
    assert module.logged["train_loss_logits"] > 0.0
    assert module.logged["train_loss_rollout"] > 0.0
