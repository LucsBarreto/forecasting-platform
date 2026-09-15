"""
Tests for NullProductRule.
"""

import pandas as pd

from src.data_validation.rules.null_product_rule import (
    NullProductRule,
)


def test_build_mask_detects_null_products() -> None:
    """Test that null products are detected."""

    dataframe = pd.DataFrame(
        {
            "COD ITEM": [10, None, 30],
        }
    )

    rule = NullProductRule()

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
            "COD ITEM": [10, None],
        }
    )

    rule = NullProductRule()

    mask = rule.build_mask(dataframe)

    assert mask.dtype == bool


def test_rule_has_expected_name() -> None:
    """Test the rule name."""

    rule = NullProductRule()

    assert rule.name == "Null Product Rule"