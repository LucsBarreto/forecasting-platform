"""
Tests for the exceptions package exports.
"""

from src.core.exceptions import (
    ApplicationError,
    ConfigurationError,
    DataError,
    DataLoadingError,
    DataValidationError,
    DirectoryNotFoundError,
    EmptyDatasetError,
    InvalidColumnTypeError,
    InvalidConfigurationError,
    InvalidDateColumnError,
    MissingColumnError,
    MissingConfigurationError,
    ModelingError,
    PipelineError,
    PipelineExecutionError,
    PredictionError,
    SchemaValidationError,
    TrainingError,
    UnsupportedFileExtensionError,
)


def test_base_exception_is_exported() -> None:
    """Test ApplicationError export."""

    assert ApplicationError is not None


def test_configuration_exceptions_are_exported() -> None:
    """Test configuration exception exports."""

    assert ConfigurationError is not None
    assert MissingConfigurationError is not None
    assert InvalidConfigurationError is not None


def test_data_exceptions_are_exported() -> None:
    """Test data exception exports."""

    assert DataError is not None
    assert DataLoadingError is not None
    assert UnsupportedFileExtensionError is not None
    assert EmptyDatasetError is not None
    assert DirectoryNotFoundError is not None


def test_modeling_exceptions_are_exported() -> None:
    """Test modeling exception exports."""

    assert ModelingError is not None
    assert TrainingError is not None
    assert PredictionError is not None


def test_pipeline_exceptions_are_exported() -> None:
    """Test pipeline exception exports."""

    assert PipelineError is not None
    assert PipelineExecutionError is not None


def test_validation_exceptions_are_exported() -> None:
    """Test validation exception exports."""

    assert DataValidationError is not None
    assert MissingColumnError is not None
    assert InvalidColumnTypeError is not None
    assert InvalidDateColumnError is not None
    assert SchemaValidationError is not None