"""
Tests for validation reports.
"""

import pandas as pd
import pytest

from src.data_validation.models import (
    ValidationReport,
    ValidationResult,
)
from src.data_validation.report import ValidationReportBuilder


def create_result(
    rule: str,
    affected_rows: int,
    removed_rows: int,
    corrected_rows: int,
    execution_time: float,
) -> ValidationResult:
    """Create a validation result for testing."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    return ValidationResult(
        rule=rule,
        total_rows_before=10,
        total_rows_after=10 - removed_rows,
        dataframe=dataframe,
        affected_rows=affected_rows,
        removed_rows=removed_rows,
        corrected_rows=corrected_rows,
        execution_time=execution_time,
        action="remove",
    )


def test_summary_returns_expected_values() -> None:
    """Test validation summary."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    results = [
        create_result(
            rule="Rule A",
            affected_rows=3,
            removed_rows=2,
            corrected_rows=0,
            execution_time=0.10,
        ),
        create_result(
            rule="Rule B",
            affected_rows=2,
            removed_rows=1,
            corrected_rows=1,
            execution_time=0.20,
        ),
    ]

    report = ValidationReport(
        dataframe=dataframe,
        results=results,
    )

    summary = ValidationReportBuilder.summary(
        report
    )

    assert summary["total_rules"] == 2
    assert summary["total_rows"] == 1
    assert summary["total_affected_rows"] == 5
    assert summary["total_removed_rows"] == 3
    assert summary["total_corrected_rows"] == 1
    assert summary["execution_time"] == pytest.approx(0.30)


def test_results_dataframe_returns_dataframe() -> None:
    """Test conversion of validation results to DataFrame."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    result = create_result(
        rule="Rule A",
        affected_rows=3,
        removed_rows=2,
        corrected_rows=0,
        execution_time=0.10,
    )

    report = ValidationReport(
        dataframe=dataframe,
        results=[result],
    )

    result_dataframe = (
        ValidationReportBuilder.results_dataframe(
            report
        )
    )

    assert isinstance(
        result_dataframe,
        pd.DataFrame,
    )

    assert len(result_dataframe) == 1
    assert result_dataframe.loc[0, "rule"] == "Rule A"
    assert result_dataframe.loc[0, "affected_rows"] == 3
    assert result_dataframe.loc[0, "removed_rows"] == 2


def test_results_dataframe_empty_report() -> None:
    """Test conversion of an empty report."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    report = ValidationReport(
        dataframe=dataframe,
        results=[],
    )

    result_dataframe = (
        ValidationReportBuilder.results_dataframe(
            report
        )
    )

    assert isinstance(
        result_dataframe,
        pd.DataFrame,
    )

    assert result_dataframe.empty
    assert "rule" in result_dataframe.columns