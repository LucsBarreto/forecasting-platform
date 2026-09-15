"""
Tests for MinimumHistoryRule.
"""

import pandas as pd

from src.data_validation.rules.minimum_history_rule import (
    MinimumHistoryRule,
)


def test_build_mask_detects_rows_outside_minimum_history() -> None:
    """Test that old records are detected."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2022-01-01",
                    "2024-01-01",
                    "2025-01-01",
                    "2026-01-01",
                ]
            ),
        }
    )

    rule = MinimumHistoryRule()

    mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        True,
        False,
        False,
        False,
    ]


def test_build_mask_returns_boolean_series() -> None:
    """Test that the mask is boolean."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2026-01-01",
                ]
            ),
        }
    )

    rule = MinimumHistoryRule()

    mask = rule.build_mask(dataframe)

    assert mask.dtype == bool


def test_build_mask_handles_invalid_dates() -> None:
    """Test that invalid dates do not become affected rows."""

    dataframe = pd.DataFrame(
        {
            "DATA": [
                "invalid",
                "2025-01-01",
                "2026-01-01",
            ],
        }
    )

    rule = MinimumHistoryRule()

    mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        False,
        False,
        False,
    ]


def test_rule_has_expected_name() -> None:
    """Test the rule name."""

    rule = MinimumHistoryRule()

    assert rule.name == "Minimum History Rule"