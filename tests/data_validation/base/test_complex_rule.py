"""
Tests for BaseComplexValidationRule.
"""

import pytest

from src.data_validation.base.complex_rule import (
    BaseComplexValidationRule,
)


def test_base_complex_validation_rule_is_abstract() -> None:
    """Test that BaseComplexValidationRule cannot be instantiated."""

    with pytest.raises(TypeError):
        BaseComplexValidationRule()