"""
regra de validação responsável por identificar retornos positivos.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=50,
    category=ValidationCategory.BUSINESS,
    enabled_by_default=True,
)
class PositiveReturnRule(BaseMaskValidationRule):
    """identifica registros com retornos positivos."""

    @property
    def name(self) -> str:
        return "Positive Return Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.BUSINESS

    @property
    def config(self):
        return settings.validation.positive_return

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """cria uma máscara para registros com retornos positivos."""

        return dataframe[
            self.config.column
        ] > 0