"""
Tests for validation categories.
"""

from src.core.enums import ValidationCategory


def test_validation_category_values() -> None:
    """Test validation category values."""

    assert ValidationCategory.STRUCTURAL == "structural"
    assert ValidationCategory.TEMPORAL == "temporal"
    assert ValidationCategory.BUSINESS == "business"
    assert ValidationCategory.STATISTICAL == "statistical"


def test_validation_category_is_string_enum() -> None:
    """Test that validation categories behave as strings."""

    assert isinstance(
        ValidationCategory.BUSINESS,
        str,
    )