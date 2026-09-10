"""
classe base para regras de validação complexas.
"""

from __future__ import annotations

from abc import abstractmethod

import pandas as pd

from src.data_validation.models import ValidationResult

from .rule import BaseValidationRule


class BaseComplexValidationRule(BaseValidationRule):
    """
    classe base para regras de validação complexas.
    """

    @abstractmethod
    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationResult:
        """
        executa a validação complexa.
        """