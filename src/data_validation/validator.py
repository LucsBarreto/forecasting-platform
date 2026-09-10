"""
engine de validação de dados.
"""

from __future__ import annotations

import pandas as pd

from src.core.logger.logger import get_logger
from src.data_validation.base import BaseValidationRule
from src.data_validation.models import ValidationReport

logger = get_logger()


class DataValidator:
    """
    executa todas as regras de validação.
    """

    def __init__(
        self,
        rules: list[BaseValidationRule],
    ) -> None:
        self._rules = rules

    @property
    def rules(self) -> list[BaseValidationRule]:
        """retorna as regras configuradas."""
        return self._rules

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationReport:
        """
        executa todas as regras sequencialmente.

        cada regra recebe o dataframe produzido pela regra anterior.
        """

        results = []

        current_dataframe = dataframe

        logger.info(
            "começando a validação dos dados"
        )

        for rule in self._rules:

            logger.info(
                f"aplicando a regra de validação: {rule.name}"
            )

            result = rule.validate(
                current_dataframe,
            )

            current_dataframe = result.dataframe

            results.append(result)

        logger.success(
            "validação completa."
        )

        return ValidationReport(
            dataframe=current_dataframe,
            results=results,
        )