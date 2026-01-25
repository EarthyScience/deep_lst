# deep_lst/losses/__init__.py
from .mse import DiagnosticMSE, RolloutMSE

__all__ = [
    "DiagnosticMSE",
    "RolloutMSE",
]
