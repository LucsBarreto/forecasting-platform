"""
regra responsável por detectar valores zero.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=90,
    category=ValidationCategory.BUSINESS,
    enabled_by_default=True,
)
class ZeroValueRule(BaseMaskValidationRule):
    """detecta valores iguais a zero."""

    @property
    def name(self) -> str:
        return "Zero Value Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.BUSINESS

    @property
    def config(self):
        return settings.validation.zero_value

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        return dataframe[self.config.column] == 0