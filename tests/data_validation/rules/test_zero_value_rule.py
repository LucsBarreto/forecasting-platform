"""
Tests for ZeroValueRule.
"""

import pandas as pd
import pytest

from src.data_validation.rules.zero_value_rule import (
    ZeroValueRule,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing zero and non-zero values."""

    return pd.DataFrame(
        {
            "VALOR": [
                100.0,
                0.0,
                250.0,
                0.0,
                -10.0,
            ],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = ZeroValueRule()

    assert rule.name == "Zero Value Rule"


def test_build_mask_detects_zero_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test that zero values are detected."""

    rule = ZeroValueRule()

    mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]


def test_build_mask_returns_boolean_series(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the result is boolean."""

    rule = ZeroValueRule()

    mask = rule.build_mask(dataframe)

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_build_mask_counts_zero_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test the number of zero values."""

    rule = ZeroValueRule()

    mask = rule.build_mask(dataframe)

    assert int(mask.sum()) == 2