"""
módulo de validação de dados.
"""

from . import rules
from .models import ValidationReport, ValidationResult
from .registry import ValidationRuleFactory

__all__ = [
    "ValidationReport",
    "ValidationResult",
    "ValidationRuleFactory",
]