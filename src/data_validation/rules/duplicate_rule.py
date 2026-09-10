"""
regra responsável por detectar registros duplicados.
"""

from __future__ import annotations

import pandas as pd

from src.config import settings
from src.core.enums import ValidationCategory
from src.data_validation.base import BaseMaskValidationRule
from src.data_validation.registry import register_rule


@register_rule(
    order=10,
    category=ValidationCategory.STRUCTURAL,
    enabled_by_default=True,
)
class DuplicateRule(BaseMaskValidationRule):
    """detecta registros duplicados."""

    @property
    def name(self) -> str:
        return "Duplicate Rule"

    @property
    def category(self) -> ValidationCategory:
        return ValidationCategory.STRUCTURAL

    @property
    def config(self):
        return settings.validation.duplicate

    @property
    def required_columns(self) -> list[str]:
        return self.config.subset or []

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:

        return dataframe.duplicated(
            subset=self.config.subset,
            keep=self.config.keep,
        )