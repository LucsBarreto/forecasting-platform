"""
regra de validação responsável por identificar clientes nulos.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=60,
    category=ValidationCategory.STRUCTURAL,
    enabled_by_default=True,
)
class NullCustomerRule(BaseMaskValidationRule):
    """identifica registros com identificadores de cliente nulos."""

    @property
    def name(self) -> str:
        return "Null Customer Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.STRUCTURAL

    @property
    def config(self):
        return settings.validation.null_customer

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """cria uma máscara para registros com clientes nulos."""

        return dataframe[
            self.config.column
        ].isna()