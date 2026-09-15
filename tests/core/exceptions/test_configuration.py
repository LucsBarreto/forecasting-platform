"""
Tests for configuration exceptions.
"""

from src.core.exceptions.configuration import (
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
)


def test_configuration_error_inherits_application_error() -> None:
    """Test the configuration exception hierarchy."""

    from src.core.exceptions.base import ApplicationError

    assert issubclass(
        ConfigurationError,
        ApplicationError,
    )


def test_missing_configuration_error_message() -> None:
    """Test the default missing configuration message."""

    error = MissingConfigurationError()

    assert str(error) == "arquivo de configuração não encontrado"


def test_invalid_configuration_error_message() -> None:
    """Test the default invalid configuration message."""

    error = InvalidConfigurationError()

    assert str(error) == "configuração invalida"


def test_configuration_errors_are_application_errors() -> None:
    """Test that configuration errors are application errors."""

    assert isinstance(
        MissingConfigurationError(),
        ConfigurationError,
    )

    assert isinstance(
        InvalidConfigurationError(),
        ConfigurationError,
    )