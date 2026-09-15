"""
Tests for base application exceptions.
"""

from src.core.exceptions.base import ApplicationError


def test_application_error_uses_default_message() -> None:
    """Test the default exception message."""

    error = ApplicationError()

    assert str(error) == "Application error."


def test_application_error_accepts_custom_message() -> None:
    """Test that a custom message is preserved."""

    error = ApplicationError(
        "custom error",
    )

    assert str(error) == "custom error"


def test_application_error_is_exception() -> None:
    """Test that ApplicationError inherits from Exception."""

    assert issubclass(
        ApplicationError,
        Exception,
    )