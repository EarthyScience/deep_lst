# deep_lst/models/resnet/__init__.py
from .loss_logic import DiagnosticMSE, RolloutMSE

__all__ = [
    "DiagnosticMSE",
    "RolloutMSE",
]