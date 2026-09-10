"""
modelos usados pelo módulo de validação de dados.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ValidationResult:
    """
    resultado da execução de uma regra de validação.
    """

    rule: str

    total_rows_before: int

    total_rows_after: int

    dataframe: pd.DataFrame

    affected_rows: int

    removed_rows: int

    corrected_rows: int

    execution_time: float

    action: str


@dataclass(slots=True)
class ValidationReport:
    """
    sumário da validação dos dados.
    """

    dataframe: pd.DataFrame

    results: list[ValidationResult]

    @property
    def total_removed_rows(self) -> int:
        """retorna o total de linhas removidas."""

        return sum(
            result.removed_rows
            for result in self.results
        )

    @property
    def total_corrected_rows(self) -> int:
        """retorna o total de linhas corrigidas."""

        return sum(
            result.corrected_rows
            for result in self.results
        )