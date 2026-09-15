"""
Tests for NullCustomerRule.
"""

import pandas as pd

from src.data_validation.rules.null_customer_rule import (
    NullCustomerRule,
)


def test_build_mask_detects_null_customers() -> None:
    """Test that null customers are detected."""

    dataframe = pd.DataFrame(
        {
            "COD CLIENTE": [1, None, 3],
        }
    )

    rule = NullCustomerRule()

    mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        False,
        True,
        False,
    ]


def test_build_mask_returns_boolean_series() -> None:
    """Test that the mask is boolean."""

    dataframe = pd.DataFrame(
        {
            "COD CLIENTE": [1, None],
        }
    )

    rule = NullCustomerRule()

    mask = rule.build_mask(dataframe)

    assert mask.dtype == bool


def test_rule_has_expected_name() -> None:
    """Test the rule name."""

    rule = NullCustomerRule()

    assert rule.name == "Null Customer Rule"