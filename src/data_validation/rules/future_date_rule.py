"""
regra de validação de datas futuras.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule
from src.data_validation.utils.columns import validate_columns


@register_rule(
    order=20,
    category=ValidationCategory.TEMPORAL,
    enabled_by_default=True,
)
class FutureDateRule(BaseMaskValidationRule):
    """detecta registros com datas futuras."""

    @property
    def name(self) -> str:
        return "Future Date Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.TEMPORAL

    @property
    def config(self):
        return settings.validation.future_date

    def _today(self) -> pd.Timestamp:
        """retorna a data atual normalizada."""

        return pd.Timestamp.today().normalize()


    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        config = self.config

        validate_columns(
            dataframe,
            self.required_columns,
        )

        today = self._today()

        if config.allow_today:
            return dataframe[
                config.column
            ] > today

        return dataframe[
            config.column
        ] >= today