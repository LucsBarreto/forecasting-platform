"""
metadados das regras de validação.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.enums import ValidationCategory
from src.data_validation.base import BaseValidationRule


@dataclass(slots=True, frozen=True)
class RuleMetadata:
    """descreve os metadados de uma regra de validação."""

    order: int

    category: ValidationCategory

    enabled_by_default: bool

    rule: type[BaseValidationRule]