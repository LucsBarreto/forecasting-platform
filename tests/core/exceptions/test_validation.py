"""
Tests for validation exceptions.
"""

from src.core.exceptions.validation import (
    DataValidationError,
    InvalidColumnTypeError,
    InvalidDateColumnError,
    MissingColumnError,
    SchemaValidationError,
)


def test_data_validation_error_inherits_application_error() -> None:
    """Test the validation exception hierarchy."""

    from src.core.exceptions.base import ApplicationError

    assert issubclass(
        DataValidationError,
        ApplicationError,
    )


def test_missing_column_error_message() -> None:
    """Test the missing column error message."""

    assert (
        str(MissingColumnError())
        == "coluna necessária não encontrada"
    )


def test_invalid_date_column_error_message() -> None:
    """Test the invalid date column message."""

    assert (
        str(InvalidDateColumnError())
        == "coluna de data invalida"
    )


def test_schema_validation_error_message() -> None:
    """Test the schema validation message."""

    assert (
        str(SchemaValidationError())
        == "validação do schema falhou"
    )


def test_invalid_column_type_error_message() -> None:
    """Test the invalid column type message."""

    error = InvalidColumnTypeError(
        column="VOLUME",
        expected="float64",
        received="object",
    )

    assert (
        str(error)
        == (
            "Column 'VOLUME' has invalid dtype. "
            "Expected 'float64', received 'object'."
        )
    )


def test_invalid_column_type_error_stores_context() -> None:
    """Test that column type error stores diagnostic information."""

    error = InvalidColumnTypeError(
        column="VOLUME",
        expected="float64",
        received="object",
    )

    assert error.column == "VOLUME"
    assert error.expected == "float64"
    assert error.received == "object"


def test_validation_errors_inherit_DATA_validation_error() -> None:
    """Test the validation exception hierarchy."""

    assert isinstance(
        MissingColumnError(),
        DataValidationError,
    )

    assert isinstance(
        InvalidDateColumnError(),
        DataValidationError,
    )

    assert isinstance(
        SchemaValidationError(),
        DataValidationError,
    )

    assert isinstance(
        InvalidColumnTypeError(
            "VOLUME",
            "float64",
            "object",
        ),
        DataValidationError,
    )