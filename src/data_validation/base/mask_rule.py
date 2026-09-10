"""
classe base para regras de validação baseadas em máscaras booleanas.
"""

from __future__ import annotations

from abc import abstractmethod
from time import perf_counter

import pandas as pd

from src.data_validation.actions import ValidationActionFactory
from src.data_validation.models import ValidationResult

from .rule import BaseValidationRule


class BaseMaskValidationRule(BaseValidationRule):
    """
    classe base para regras baseadas em máscaras booleanas.
    """

    @property
    @abstractmethod
    def config(self):
        """retorna a configuração da regra."""

    @property
    @abstractmethod
    def name(self) -> str:
        """retorna o nome da regra."""

    @property
    @abstractmethod
    def category(self):
        """retorna a categoria da regra."""

    @property
    def action(self):
        """retorna a ação configurada para a regra."""

        if isinstance(self.config, dict):
            return self.config["action"]

        return self.config.action

    @property
    def required_columns(self) -> list[str]:
        """retorna as colunas necessárias para a regra."""

        if isinstance(self.config, dict):
            column = self.config.get("column")

            if column is None:
                return []

            return [column]

        column = getattr(self.config, "column", None)

        if column is None:
            return []

        return [column]

    @abstractmethod
    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        """
        cria a máscara booleana da regra.
        """

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> ValidationResult:
        """
        executa a validação da regra.
        """

        start = perf_counter()

        config = self.config

        if isinstance(config, dict):
            enabled = config.get("enabled", True)
        else:
            enabled = getattr(config, "enabled", True)

        if not enabled:
            return ValidationResult(
                rule=self.name,
                dataframe=dataframe,
                total_rows_before=len(dataframe),
                total_rows_after=len(dataframe),
                affected_rows=0,
                removed_rows=0,
                corrected_rows=0,
                execution_time=0.0,
                action="skipped",
            )

        mask = self.build_mask(dataframe)

        affected_rows = int(mask.sum())

        strategy = ValidationActionFactory.create(
            self.action
        )

        result = strategy.execute(
            dataframe=dataframe,
            mask=mask,
        )

        execution_time = perf_counter() - start

        return ValidationResult(
            rule=self.name,
            dataframe=result.dataframe,
            total_rows_before=len(dataframe),
            total_rows_after=len(result.dataframe),
            affected_rows=affected_rows,
            removed_rows=result.removed_rows,
            corrected_rows=result.corrected_rows,
            execution_time=execution_time,
            action=result.action,
        )