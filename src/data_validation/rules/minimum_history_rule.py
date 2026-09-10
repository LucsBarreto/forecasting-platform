"""
regra de validação responsável por validar o histórico mínimo de dados.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=90,
    category=ValidationCategory.TEMPORAL,
    enabled_by_default=True,
)
class MinimumHistoryRule(BaseMaskValidationRule):
    """identifica registros fora da janela mínima de histórico configurada."""

    @property
    def name(self) -> str:
        return "Minimum History Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.TEMPORAL

    @property
    def config(self):
        return settings.validation.minimum_history

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """cria uma máscara para registros anteriores ao período mínimo de histórico."""

        config = self.config

        dates = pd.to_datetime(
            dataframe[config.column],
            errors="coerce",
            format="mixed",
        )

        latest_date = dates.max()

        if pd.isna(latest_date):
            return pd.Series(
                False,
                index=dataframe.index,
                dtype=bool,
            )

        minimum_date = latest_date - pd.DateOffset(
            years=config.minimum_years,
        )

        return dates < minimum_date