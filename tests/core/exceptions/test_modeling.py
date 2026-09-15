"""
Tests for modeling exceptions.
"""

from src.core.exceptions.modeling import (
    ModelingError,
    PredictionError,
    TrainingError,
)


def test_modeling_error_inherits_application_error() -> None:
    """Test the modeling exception hierarchy."""

    from src.core.exceptions.base import ApplicationError

    assert issubclass(
        ModelingError,
        ApplicationError,
    )


def test_training_error_message() -> None:
    """Test the default training error message."""

    assert (
        str(TrainingError())
        == "treinamento do modelo falhou"
    )


def test_prediction_error_message() -> None:
    """Test the default prediction error message."""

    assert (
        str(PredictionError())
        == "predição falhou"
    )


def test_modeling_errors_inherit_modeling_error() -> None:
    """Test the modeling exception hierarchy."""

    assert isinstance(
        TrainingError(),
        ModelingError,
    )

    assert isinstance(
        PredictionError(),
        ModelingError,
    )