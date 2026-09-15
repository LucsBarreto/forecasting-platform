"""
Tests for FrequencyType enum.
"""

from src.core.enums.frequency import FrequencyType


def test_frequency_type_values() -> None:
    """Test supported frequency values."""

    assert FrequencyType.DAILY == "D"
    assert FrequencyType.WEEKLY == "W"
    assert FrequencyType.MONTHLY == "M"
    assert FrequencyType.QUARTERLY == "Q"
    assert FrequencyType.YEARLY == "Y"


def test_frequency_type_is_str_enum() -> None:
    """Test that FrequencyType members behave as strings."""

    assert isinstance(FrequencyType.DAILY, str)


def test_frequency_type_contains_expected_members() -> None:
    """Test all expected members exist."""

    assert set(FrequencyType) == {
        FrequencyType.DAILY,
        FrequencyType.WEEKLY,
        FrequencyType.MONTHLY,
        FrequencyType.QUARTERLY,
        FrequencyType.YEARLY,
    }