"""
Tests for FutureDateRule.
"""

import pandas as pd
import pytest

from unittest.mock import patch

from src.data_validation.rules.future_date_rule import FutureDateRule


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing past, today and future dates."""

    today = pd.Timestamp("2026-08-18")

    return pd.DataFrame(
        {
            "DATA": [
                today - pd.DateOffset(days=2),
                today,
                today + pd.DateOffset(days=2),
            ],
            "VOLUME": [100, 200, 300],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = FutureDateRule()

    assert rule.name == "Future Date Rule"


def test_category() -> None:
    """Test the rule category."""

    from src.core.enums import ValidationCategory

    rule = FutureDateRule()

    assert rule.category == ValidationCategory.TEMPORAL


def test_build_mask_detects_future_dates(
    dataframe: pd.DataFrame,
) -> None:
    """Test that future dates are detected."""

    rule = FutureDateRule()

    with patch.object(
        rule,
        "_today",
        return_value=pd.Timestamp("2026-08-18"),
    ):
        mask = rule.build_mask(dataframe)

    assert mask.tolist() == [
        False,
        False,
        True,
    ]


def test_build_mask_returns_boolean_series(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the mask is boolean."""

    rule = FutureDateRule()

    mask = rule.build_mask(dataframe)

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool