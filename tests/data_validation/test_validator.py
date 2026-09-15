"""
Tests for the data validation engine.
"""

from unittest.mock import MagicMock

import pandas as pd

from src.data_validation.models import ValidationReport
from src.data_validation.validator import DataValidator


def test_validate_returns_validation_report() -> None:
    """Test that validation returns a ValidationReport."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100, 200],
        }
    )

    rule = MagicMock()
    rule.name = "Test Rule"

    result = MagicMock()
    result.dataframe = dataframe

    rule.validate.return_value = result

    validator = DataValidator(
        rules=[rule],
    )

    report = validator.validate(dataframe)

    assert isinstance(
        report,
        ValidationReport,
    )


def test_validate_calls_each_rule() -> None:
    """Test that every configured rule is executed."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100, 200],
        }
    )

    first_rule = MagicMock()
    first_rule.name = "First Rule"

    second_rule = MagicMock()
    second_rule.name = "Second Rule"

    first_result = MagicMock()
    first_result.dataframe = dataframe

    second_result = MagicMock()
    second_result.dataframe = dataframe

    first_rule.validate.return_value = first_result
    second_rule.validate.return_value = second_result

    validator = DataValidator(
        rules=[
            first_rule,
            second_rule,
        ]
    )

    validator.validate(dataframe)

    first_rule.validate.assert_called_once_with(dataframe)
    second_rule.validate.assert_called_once_with(dataframe)


def test_validate_passes_result_dataframe_to_next_rule() -> None:
    """Test that rules execute sequentially."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100, 200],
        }
    )

    cleaned_dataframe = pd.DataFrame(
        {
            "VOLUME": [100],
        }
    )

    first_rule = MagicMock()
    first_rule.name = "First Rule"

    second_rule = MagicMock()
    second_rule.name = "Second Rule"

    first_result = MagicMock()
    first_result.dataframe = cleaned_dataframe

    second_result = MagicMock()
    second_result.dataframe = cleaned_dataframe

    first_rule.validate.return_value = first_result
    second_rule.validate.return_value = second_result

    validator = DataValidator(
        rules=[
            first_rule,
            second_rule,
        ]
    )

    validator.validate(dataframe)

    first_rule.validate.assert_called_once_with(dataframe)
    second_rule.validate.assert_called_once_with(
        cleaned_dataframe
    )


def test_validate_collects_all_results() -> None:
    """Test that all validation results are collected."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100, 200],
        }
    )

    first_rule = MagicMock()
    first_rule.name = "First Rule"

    second_rule = MagicMock()
    second_rule.name = "Second Rule"

    first_result = MagicMock()
    first_result.dataframe = dataframe

    second_result = MagicMock()
    second_result.dataframe = dataframe

    first_rule.validate.return_value = first_result
    second_rule.validate.return_value = second_result

    validator = DataValidator(
        rules=[
            first_rule,
            second_rule,
        ]
    )

    report = validator.validate(dataframe)

    assert report.results == [
        first_result,
        second_result,
    ]


def test_validate_with_no_rules() -> None:
    """Test validation with no configured rules."""

    dataframe = pd.DataFrame(
        {
            "VOLUME": [100, 200],
        }
    )

    validator = DataValidator(
        rules=[],
    )

    report = validator.validate(dataframe)

    assert isinstance(
        report,
        ValidationReport,
    )

    pd.testing.assert_frame_equal(
        report.dataframe,
        dataframe,
    )

    assert report.results == []