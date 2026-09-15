"""
Tests for BaseMaskValidationRule.
"""

import pandas as pd

from src.core.enums import ValidationActionType
from src.data_validation.base.mask_rule import (
    BaseMaskValidationRule,
)


class TestMaskRule(BaseMaskValidationRule):
    """Concrete rule used for testing."""

    @property
    def name(self) -> str:
        return "test_rule"

    @property
    def category(self) -> str:
        return "test"

    @property
    def config(self):
        return {}

    @property
    def action(self) -> ValidationActionType:
        return ValidationActionType.REMOVE

    def build_mask(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.Series:
        return dataframe["VALOR"] <= 0


def test_mask_rule_builds_validation_result() -> None:
    """Test validation result generation."""

    dataframe = pd.DataFrame(
        {
            "VALOR": [100, 0, 200, -10],
        }
    )

    result = TestMaskRule().validate(dataframe)

    assert result.rule == "test_rule"
    assert result.total_rows_before == 4
    assert result.total_rows_after == 2
    assert result.affected_rows == 2
    assert result.removed_rows == 2
    assert result.corrected_rows == 0
    assert result.action == "remove"

    assert result.dataframe["VALOR"].tolist() == [
        100,
        200,
    ]


def test_mask_rule_preserves_dataframe_when_no_rows_are_affected() -> None:
    """Test validation when no rows match the mask."""

    dataframe = pd.DataFrame(
        {
            "VALOR": [100, 200, 300],
        }
    )

    result = TestMaskRule().validate(dataframe)

    assert result.affected_rows == 0
    assert result.removed_rows == 0
    assert len(result.dataframe) == 3