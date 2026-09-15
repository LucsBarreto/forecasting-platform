"""
Tests for ValidationActionType enum.
"""

from src.core.enums.validation import ValidationActionType


def test_validation_action_type_values() -> None:
    """Test supported validation actions."""

    assert ValidationActionType.REMOVE == "remove"
    assert ValidationActionType.KEEP == "keep"
    assert ValidationActionType.WARN == "warn"


def test_validation_action_type_is_str_enum() -> None:
    """Test that members behave as strings."""

    assert isinstance(
        ValidationActionType.REMOVE,
        str,
    )


def test_validation_action_type_contains_expected_members() -> None:
    """Test all expected members exist."""

    assert set(ValidationActionType) == {
        ValidationActionType.REMOVE,
        ValidationActionType.KEEP,
        ValidationActionType.WARN,
    }