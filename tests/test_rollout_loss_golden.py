"""Pins RolloutLoss to its values before the objective refactor.

tests/fixtures/rollout_loss_v1.json holds, for every case in CASES, the total loss, the logged
components and the gradients of the stub model's parameters, computed with the RolloutLoss of
develop 5274122 (legacy keyword arguments, loss function read from config["model"]["loss_fn"]).
Never regenerate the fixture from refactored code.

Each case names the config overrides that say today what the legacy keyword arguments said
then; the objective is built the way training builds it, from the composed `objective` node.

The stub model's output depends on the last window slot, on the hidden (NaN) slots, on the
context window and on the step forcings, so feeding the model a different window or forcing
at any lead changes the loss. The gradients separate detach_rollout on from off.
"""

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from deep_lst.lit_module import build_objective

CONF_DIR = Path(__file__).resolve().parents[1] / "conf"
FIXTURE = Path(__file__).parent / "fixtures" / "rollout_loss_v1.json"

B = 4
LOOKBACK = 4
HORIZON = 3
T = LOOKBACK + HORIZON + 1
N_FORCING = 2
N_CONTEXT = 1
SEED = 7  # torch.rand of the init_fill hide mask

LST = ("seviri", "LST")
MLE = ("seviri", "MLE")

# Per case: the config overrides on top of the defaults (objective=rollout loss=smoothl1) and
# the run-time keys; n_states and joint describe the batch. joint = the second state channel
# is the diagnostic copy of the prognostic. The legacy keyword arguments each case was pinned
# with are given where they differ from detach_rollout=False, teacher_forcing_loss=True,
# incremental_loss=False, rollout_loss=True, loss_fn=smooth_l1.
CASES = {
    "default": {
        "n_states": 1,
        "overrides": [],
        "prognostic_keys": [LST],
    },
    # RolloutLoss(prognostic_keys=...) alone: a legacy config that omitted detach_rollout trained
    # detached (code default True) although the yaml default was false
    "legacy_code_defaults": {
        "n_states": 1,
        "overrides": ["objective.detach_rollout=true"],
        "prognostic_keys": [LST],
    },
    # detach_rollout=True
    "detach_rollout": {
        "n_states": 1,
        "overrides": ["objective.detach_rollout=true"],
        "prognostic_keys": [LST],
    },
    # incremental_loss=True
    "increment": {
        "n_states": 1,
        "overrides": ["objective.terms.increment=1"],
        "prognostic_keys": [LST],
        "inc_variances": {"LST": 0.25},
    },
    # incremental_loss=True, teacher_forcing_loss=False
    "increment_without_teacher_forcing": {
        "n_states": 1,
        "overrides": ["objective.terms.increment=1", "objective.terms.teacher_forcing=0"],
        "prognostic_keys": [LST],
        "inc_variances": {"LST": 0.25},
    },
    # rollout_loss=False
    "teacher_forcing_only": {
        "n_states": 1,
        "overrides": ["objective.terms.rollout=0"],
        "prognostic_keys": [LST],
    },
    # teacher_forcing_loss=False
    "rollout_only": {
        "n_states": 1,
        "overrides": ["objective.terms.teacher_forcing=0"],
        "prognostic_keys": [LST],
    },
    # loss_fn=mse
    "mse": {
        "n_states": 1,
        "overrides": ["loss=mse"],
        "prognostic_keys": [LST],
    },
    # loss_fn=mae
    "mae": {
        "n_states": 1,
        "overrides": ["loss=mae"],
        "prognostic_keys": [LST],
    },
    # no prognostic (data=full_lstdiag); pinned with rollout_loss=True, which the missing
    # prognostic state switched off
    "diagnostic_only": {
        "n_states": 1,
        "overrides": ["objective=diagnostic"],
        "prognostic_keys": [],
        "diagnostic_keys": [LST],
    },
    "extra_diagnostic": {
        "n_states": 2,
        "overrides": [],
        "prognostic_keys": [LST],
        "diagnostic_keys": [MLE],
    },
    # init_fill=none
    "joint_init_fill_none": {
        "n_states": 2,
        "joint": True,
        "overrides": [],
        "prognostic_keys": [LST],
        "diagnostic_keys": [LST],
    },
    # init_fill=gated
    "joint_init_fill_gated": {
        "n_states": 2,
        "joint": True,
        "overrides": ["objective=rollout_gated"],
        "prognostic_keys": [LST],
        "diagnostic_keys": [LST],
    },
    # init_fill=gated, detach_rollout=True
    "joint_init_fill_gated_detach": {
        "n_states": 2,
        "joint": True,
        "overrides": ["objective=rollout_gated", "objective.detach_rollout=true"],
        "prognostic_keys": [LST],
        "diagnostic_keys": [LST],
    },
    # init_fill=always
    "joint_init_fill_always": {
        "n_states": 2,
        "joint": True,
        "overrides": ["objective=rollout_always"],
        "prognostic_keys": [LST],
        "diagnostic_keys": [LST],
    },
}


class _InputSensitiveModel(torch.nn.Module):
    """One value per output channel that moves with every input the loss hands over."""

    def __init__(self, n_out):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(0.1, 0.5, n_out))
        self.scale = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.1, 0.4]))

    def forward(self, state_window, X_dyn_window, step_X_dyn):
        # state_window (N, L, H, W, nS), X_dyn_window (N, L, H, W, nFc), step_X_dyn (N, H, W, nF)
        last_state = torch.nan_to_num(state_window[:, -1]).sum(dim=-1, keepdim=True)
        hidden_share = torch.isnan(state_window).float().mean(dim=(1, -1)).unsqueeze(-1)
        context_mean = X_dyn_window.mean(dim=(1, -1)).unsqueeze(-1)
        step_sum = step_X_dyn.sum(dim=-1, keepdim=True)
        signal = (
            self.scale[0] * last_state
            + self.scale[1] * hidden_share
            + self.scale[2] * context_mean
            + self.scale[3] * step_sum
        )
        return signal + self.bias


class _StubModule:
    """The LitModelModule attributes RolloutLoss reads, plus a log() that records values."""

    def __init__(self, model):
        self.model = model
        self.config = {"window": {"lookback": LOOKBACK, "horizon": HORIZON}}
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value.detach())


def _batch(n_states, joint):
    generator = torch.Generator().manual_seed(0)
    y_prog = torch.randn(B, T, 1, 1, n_states, generator=generator)
    y_prog[0, LOOKBACK + 1, 0, 0, 0] = float("nan")  # one masked target
    y_prog[1, LOOKBACK, 0, 0, 0] = float("nan")  # one init without an observation
    y_prog[2, 1, 0, 0, 0] = float("nan")  # one gap inside the lookback
    if joint:
        y_prog[..., 1] = y_prog[..., 0]
    X_dyn = torch.randn(B, T, 1, 1, N_FORCING, generator=generator)
    X_dyn_ctx = torch.randn(B, T, 1, 1, N_CONTEXT, generator=generator)
    return {"y_prog": y_prog, "X_dyn": X_dyn, "X_dyn_ctx": X_dyn_ctx}


def _objective(overrides, **runtime_keys):
    """The objective as LitModelModule builds it, from the composed `objective` node."""
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    objective_config = OmegaConf.to_container(cfg.objective, resolve=True)
    return build_objective(objective_config, **runtime_keys)


def _evaluate(case):
    """Total loss, logged components and parameter gradients of one case."""
    runtime_keys = dict(case)
    n_states = runtime_keys.pop("n_states")
    joint = runtime_keys.pop("joint", False)
    overrides = runtime_keys.pop("overrides")

    model = _InputSensitiveModel(n_out=n_states)
    module = _StubModule(model)
    objective = _objective(overrides, **runtime_keys)

    torch.manual_seed(SEED)
    total = objective(module, _batch(n_states, joint), stage="train")
    total.backward()

    return {
        "total": float(total.detach()),
        "logged": module.logged,
        "grad_bias": model.bias.grad.tolist(),
        "grad_scale": model.scale.grad.tolist(),
    }


@pytest.mark.parametrize("name", list(CASES))
def test_rollout_loss_matches_the_values_before_the_refactor(name):
    expected = json.loads(FIXTURE.read_text())[name]

    result = _evaluate(CASES[name])

    assert result["total"] == pytest.approx(expected["total"], rel=1e-5, abs=1e-7)
    assert result["logged"] == pytest.approx(expected["logged"], rel=1e-5, abs=1e-7)
    assert result["grad_bias"] == pytest.approx(expected["grad_bias"], rel=1e-5, abs=1e-7)
    assert result["grad_scale"] == pytest.approx(expected["grad_scale"], rel=1e-5, abs=1e-7)


def test_fixture_and_cases_cover_each_other():
    assert sorted(json.loads(FIXTURE.read_text())) == sorted(CASES)


def test_rollout_objective_without_prognostics_equals_the_diagnostic_objective():
    """data=full_lstdiag trained through objective=rollout before objective=diagnostic existed:
    without a prognostic state the rollout branch is empty either way."""
    expected = json.loads(FIXTURE.read_text())["diagnostic_only"]
    case = dict(CASES["diagnostic_only"], overrides=["objective=rollout"])

    result = _evaluate(case)

    assert result["total"] == pytest.approx(expected["total"], rel=1e-5, abs=1e-7)
    assert result["logged"] == pytest.approx(expected["logged"], rel=1e-5, abs=1e-7)
