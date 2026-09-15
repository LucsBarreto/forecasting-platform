"""
Tests for data exceptions.
"""

from src.core.exceptions.data import (
    DataError,
    DataLoadingError,
    DirectoryNotFoundError,
    EmptyDatasetError,
    UnsupportedFileExtensionError,
)


def test_data_error_inherits_application_error() -> None:
    """Test the data exception hierarchy."""

    from src.core.exceptions.base import ApplicationError

    assert issubclass(
        DataError,
        ApplicationError,
    )


def test_data_loading_error_message() -> None:
    """Test the default data loading message."""

    assert (
        str(DataLoadingError())
        == "dataset não foi carregado"
    )


def test_unsupported_file_extension_error_message() -> None:
    """Test the default unsupported extension message."""

    assert (
        str(UnsupportedFileExtensionError())
        == "extensão de arquivo não suportada"
    )


def test_empty_dataset_error_message() -> None:
    """Test the default empty dataset message."""

    assert (
        str(EmptyDatasetError())
        == "dataset esta vazio"
    )


def test_directory_not_found_error_message() -> None:
    """Test the default directory message."""

    assert (
        str(DirectoryNotFoundError())
        == "pasta não encontrada"
    )


def test_data_errors_inherit_DATA_error() -> None:
    """Test the data exception hierarchy."""

    assert isinstance(
        DataLoadingError(),
        DataError,
    )

    assert isinstance(
        UnsupportedFileExtensionError(),
        DataError,
    )

    assert isinstance(
        EmptyDatasetError(),
        DataError,
    )

    assert isinstance(
        DirectoryNotFoundError(),
        DataError,
    )