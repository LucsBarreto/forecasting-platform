"""
ações de validação de dados
"""

from .factory import ValidationActionFactory

# importa ações para registrar automaticamente
from .keep import KeepAction
from .remove import RemoveAction
from .warn import WarnAction

__all__ = [
    "ValidationActionFactory",
    "KeepAction",
    "RemoveAction",
    "WarnAction",
]