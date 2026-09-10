"""
relatórios de validação de dados.
"""

from __future__ import annotations

import pandas as pd

from src.data_validation.models import (
    ValidationReport,
    ValidationResult,
)


class ValidationReportBuilder:
    """
    constrói informações resumidas a partir de um validationreport.
    """

    @staticmethod
    def summary(
        report: ValidationReport,
    ) -> dict[str, object]:
        """
        retorna um resumo da validação.
        """

        return {
            "total_rules": len(report.results),
            "total_rows": len(report.dataframe),
            "total_affected_rows": sum(
                result.affected_rows
                for result in report.results
            ),
            "total_removed_rows": report.total_removed_rows,
            "total_corrected_rows": report.total_corrected_rows,
            "execution_time": sum(
                result.execution_time
                for result in report.results
            ),
        }

    @staticmethod
    def results_dataframe(
        report: ValidationReport,
    ) -> pd.DataFrame:
        """
        converte os resultados das regras para dataframe.
        """

        if not report.results:
            return pd.DataFrame(
                columns=[
                    "rule",
                    "total_rows_before",
                    "total_rows_after",
                    "affected_rows",
                    "removed_rows",
                    "corrected_rows",
                    "execution_time",
                    "action",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "rule": result.rule,
                    "total_rows_before": result.total_rows_before,
                    "total_rows_after": result.total_rows_after,
                    "affected_rows": result.affected_rows,
                    "removed_rows": result.removed_rows,
                    "corrected_rows": result.corrected_rows,
                    "execution_time": result.execution_time,
                    "action": result.action,
                }
                for result in report.results
            ]
        )