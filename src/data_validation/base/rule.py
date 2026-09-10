"""
regra base de validação.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.data_validation.models import ValidationResult


class BaseValidationRule(ABC):
    """
    classe base para cada regra de validação.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """nome da regra."""

    @property
    @abstractmethod
    def category(self) -> str:
        """categoria da regra."""

    @abstractmethod
    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationResult:
        """
        executa a validação.
        """