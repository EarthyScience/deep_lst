# deep_lst/vit/__init__.py
# NOTE: the vit path has NOT been migrated to the packed-tensor model/loss. RolloutLoss is
# re-exported so the package imports, but vit training/inference needs its own migration
# (vit/model.py still takes a TensorDict).
from deep_lst.losses import DiagnosticMSE, RolloutLoss

__all__ = [
    "DiagnosticMSE",
    "RolloutLoss",
]
