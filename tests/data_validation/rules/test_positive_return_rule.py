"""
Tests for positive return rule.
"""

import pandas as pd
import pytest

from src.data_validation.rules.positive_return_rule import (
    PositiveReturnRule,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing positive and non-positive returns."""

    return pd.DataFrame(
        {
            "DEVOLUCAO": [
                100.0,
                0.0,
                -50.0,
                25.0,
            ],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = PositiveReturnRule()

    assert rule.name == "Positive Return Rule"


def test_build_mask_detects_positive_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test that positive values are detected."""

    rule = PositiveReturnRule()

    mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_build_mask_returns_boolean_series(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the result is boolean."""

    rule = PositiveReturnRule()

    mask = rule.build_mask(dataframe)

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_build_mask_counts_positive_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test the number of affected rows."""

    rule = PositiveReturnRule()

    mask = rule.build_mask(dataframe)

    assert int(mask.sum()) == 2