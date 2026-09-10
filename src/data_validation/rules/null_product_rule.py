"""
regra de validação responsável por identificar produtos nulos.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=70,
    category=ValidationCategory.STRUCTURAL,
    enabled_by_default=True,
)
class NullProductRule(BaseMaskValidationRule):
    """identifica registros com identificadores de produto nulos."""

    @property
    def name(self) -> str:
        return "Null Product Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.STRUCTURAL

    @property
    def config(self):
        return settings.validation.null_product

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """cria uma máscara para registros com produtos nulos."""

        return dataframe[
            self.config.column
        ].isna()