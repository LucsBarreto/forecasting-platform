"""
Tests for DuplicateRule.
"""

import pandas as pd
import pytest

from src.data_validation.rules.duplicate_rule import DuplicateRule


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a dataframe containing duplicated rows."""

    return pd.DataFrame(
        {
            "COD CLIENTE": [1, 1, 2, 3],
            "COD ITEM": [10, 10, 20, 30],
            "VOLUME": [100, 100, 200, 300],
        }
    )


def test_rule_name() -> None:
    """Test the rule name."""

    rule = DuplicateRule()

    assert rule.name == "Duplicate Rule"


def test_validate_removes_duplicates(
    dataframe: pd.DataFrame,
) -> None:
    """Test that duplicated rows are removed."""

    rule = DuplicateRule()

    result = rule.validate(dataframe)

    assert result.removed_rows == 1
    assert result.affected_rows == 1
    assert result.total_rows_before == 4
    assert result.total_rows_after == 3


def test_validate_preserves_first_duplicate(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the first duplicated row is preserved."""

    rule = DuplicateRule()

    result = rule.validate(dataframe)

    assert len(result.dataframe) == 3
    assert result.dataframe.iloc[0]["COD CLIENTE"] == 1


def test_validate_does_not_modify_original_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the original dataframe remains unchanged."""

    original = dataframe.copy()

    rule = DuplicateRule()
    rule.validate(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_validate_returns_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Test that validation returns a dataframe."""

    rule = DuplicateRule()

    result = rule.validate(dataframe)

    assert isinstance(result.dataframe, pd.DataFrame)