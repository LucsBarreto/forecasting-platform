"""
Tests for BaseValidationRule.
"""

import pytest

from src.data_validation.base.rule import BaseValidationRule


def test_base_validation_rule_is_abstract() -> None:
    """Test that BaseValidationRule cannot be instantiated."""

    with pytest.raises(TypeError):
        BaseValidationRule()