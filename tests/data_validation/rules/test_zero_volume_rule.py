"""
Tests for ZeroVolumeRule.
"""

import pandas as pd
import pytest

from src.data_validation.rules.zero_volume_rule import (
    ZeroVolumeRule,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing zero and non-zero volumes."""

    return pd.DataFrame(
        {
            "VOLUME": [
                100.0,
                0.0,
                200.0,
                0.0,
                300.0,
            ],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = ZeroVolumeRule()

    assert rule.name == "Zero Volume Rule"


def test_category() -> None:
    """Test the rule category."""

    from src.core.enums import ValidationCategory

    rule = ZeroVolumeRule()

    assert rule.category == ValidationCategory.BUSINESS


def test_build_mask_detects_zero_volume(
    dataframe: pd.DataFrame,
) -> None:
    """Test that zero volume is detected."""

    rule = ZeroVolumeRule()

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

    rule = ZeroVolumeRule()

    mask = rule.build_mask(dataframe)

    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool


def test_build_mask_counts_zero_volume(
    dataframe: pd.DataFrame,
) -> None:
    """Test the number of zero-volume rows."""

    rule = ZeroVolumeRule()

    mask = rule.build_mask(dataframe)

    assert int(mask.sum()) == 2