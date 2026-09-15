"""
Tests for data validation models.
"""

import pandas as pd

from src.data_validation.models import (
    ValidationReport,
    ValidationResult,
)


def test_validation_result_stores_values() -> None:
    """Test that ValidationResult stores validation data."""

    dataframe = pd.DataFrame(
        {"VALOR": [100, 200]},
    )

    result = ValidationResult(
        rule="test_rule",
        total_rows_before=3,
        total_rows_after=2,
        dataframe=dataframe,
        affected_rows=1,
        removed_rows=1,
        corrected_rows=0,
        execution_time=0.01,
        action="remove",
    )

    assert result.rule == "test_rule"
    assert result.total_rows_before == 3
    assert result.total_rows_after == 2
    assert result.affected_rows == 1
    assert result.removed_rows == 1
    assert result.corrected_rows == 0
    assert result.action == "remove"


def test_validation_report_calculates_total_removed_rows() -> None:
    """Test total removed rows."""

    dataframe = pd.DataFrame()

    results = [
        ValidationResult(
            rule="rule_1",
            total_rows_before=10,
            total_rows_after=8,
            dataframe=dataframe,
            affected_rows=2,
            removed_rows=2,
            corrected_rows=0,
            execution_time=0.01,
            action="remove",
        ),
        ValidationResult(
            rule="rule_2",
            total_rows_before=8,
            total_rows_after=7,
            dataframe=dataframe,
            affected_rows=1,
            removed_rows=1,
            corrected_rows=0,
            execution_time=0.01,
            action="remove",
        ),
    ]

    report = ValidationReport(
        dataframe=dataframe,
        results=results,
    )

    assert report.total_removed_rows == 3


def test_validation_report_calculates_total_corrected_rows() -> None:
    """Test total corrected rows."""

    dataframe = pd.DataFrame()

    results = [
        ValidationResult(
            rule="rule_1",
            total_rows_before=10,
            total_rows_after=10,
            dataframe=dataframe,
            affected_rows=2,
            removed_rows=0,
            corrected_rows=2,
            execution_time=0.01,
            action="keep",
        ),
        ValidationResult(
            rule="rule_2",
            total_rows_before=10,
            total_rows_after=10,
            dataframe=dataframe,
            affected_rows=3,
            removed_rows=0,
            corrected_rows=3,
            execution_time=0.01,
            action="keep",
        ),
    ]

    report = ValidationReport(
        dataframe=dataframe,
        results=results,
    )

    assert report.total_corrected_rows == 5