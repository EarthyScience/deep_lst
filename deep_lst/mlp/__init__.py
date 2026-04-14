# deep_lst/mlp/__init__.py
from deep_lst.losses import DiagnosticCRPS, DiagnosticMSE, RolloutCRPS, RolloutMSE

__all__ = [
    "DiagnosticMSE",
    "RolloutMSE",
    "DiagnosticCRPS",
    "RolloutCRPS",
]
