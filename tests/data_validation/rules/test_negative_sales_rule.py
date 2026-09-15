"""
Tests for NegativeSalesRule.
"""

import pandas as pd
import pytest

from src.data_validation.rules.negative_sales_rule import (
    NegativeSalesRule,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing negative and positive sales."""

    return pd.DataFrame(
        {
            "VALOR": [
                100.0,
                -50.0,
                0.0,
                -10.0,
                200.0,
            ],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = NegativeSalesRule()

    assert rule.name == "Negative Sales Rule"


def test_build_mask_detects_negative_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test that negative values are detected."""

    rule = NegativeSalesRule()

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
    """Test that the result is a boolean series."""

    rule = NegativeSalesRule()

    mask = rule.build_mask(dataframe)

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_build_mask_counts_negative_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test the number of affected rows."""

    rule = NegativeSalesRule()

    mask = rule.build_mask(dataframe)

    assert int(mask.sum()) == 2