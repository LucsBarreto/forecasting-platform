"""
regra de validação para vendas negativas.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import (
    ValidationActionType,
    ValidationCategory,
)
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=40,
    category=ValidationCategory.BUSINESS,
    enabled_by_default=True,
)
class NegativeSalesRule(BaseMaskValidationRule):
    """detecta registros com vendas negativas."""

    @property
    def name(self) -> str:
        return "Negative Sales Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.BUSINESS

    @property
    def config(self):
        return settings.validation.negative_sales

    @property
    def action(self) -> ValidationActionType:
        return self.config.action

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        return dataframe[self.config.column] < 0