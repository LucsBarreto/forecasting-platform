"""
Tests for pipeline exceptions.
"""

from src.core.exceptions.pipeline import (
    PipelineError,
    PipelineExecutionError,
)


def test_pipeline_error_inherits_application_error() -> None:
    """Test the pipeline exception hierarchy."""

    from src.core.exceptions.base import ApplicationError

    assert issubclass(
        PipelineError,
        ApplicationError,
    )


def test_pipeline_execution_error_message() -> None:
    """Test the default pipeline execution message."""

    assert (
        str(PipelineExecutionError())
        == "execução do pipeline falhou"
    )


def test_pipeline_execution_error_inherits_pipeline_error() -> None:
    """Test the pipeline exception hierarchy."""

    assert isinstance(
        PipelineExecutionError(),
        PipelineError,
    )