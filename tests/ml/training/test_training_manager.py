"""
Tests for TrainingManager.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ml.models.base_model import BaseModel
from src.ml.training.model_selection import ModelSelector
from src.ml.training.training_manager import (
    TrainingManager,
    TrainingResult,
)


class DummyModel(BaseModel):
    """Dummy model used for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            0,
            index=X.index,
            dtype=float,
        )


@pytest.fixture
def features() -> pd.DataFrame:
    """Return training features."""

    return pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [5, 4, 3, 2, 1],
        }
    )


@pytest.fixture
def target() -> pd.Series:
    """Return training target."""

    return pd.Series(
        [10, 20, 30, 40, 50],
        name="target",
    )


@pytest.fixture
def selector() -> ModelSelector:
    """Return a model selector."""

    return ModelSelector()


@pytest.fixture
def manager(
    selector: ModelSelector,
) -> TrainingManager:
    """Return a training manager."""

    return TrainingManager(
        selector=selector,
    )


def test_training_result_contains_models(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test training result type."""

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=DummyModel(),
        ):
            result = manager.train(
                features,
                target,
            )

    assert isinstance(
        result,
        TrainingResult,
    )

    assert "dummy" in result.models


def test_training_returns_fitted_models(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test fitted model creation."""

    model = DummyModel()

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=model,
        ):
            result = manager.train(
                features,
                target,
            )

    assert result.models["dummy"] is model


def test_all_selected_models_are_trained(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test training of multiple selected models."""

    model_names = (
        "model_a",
        "model_b",
        "model_c",
    )

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=model_names,
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            side_effect=[
                DummyModel(),
                DummyModel(),
                DummyModel(),
            ],
        ) as factory:
            result = manager.train(
                features,
                target,
            )

    assert set(result.models) == set(
        model_names
    )

    assert factory.call_count == 3


def test_model_specific_parameters_are_forwarded(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test model-specific parameter forwarding."""

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=DummyModel(),
        ) as factory:
            manager.train(
                features,
                target,
                model_params={
                    "dummy": {
                        "example": 123,
                    }
                },
            )

    factory.assert_called_once_with(
        "dummy",
        example=123,
    )


def test_missing_model_parameters_are_allowed(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test training without model-specific parameters."""

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=DummyModel(),
        ) as factory:
            manager.train(
                features,
                target,
            )

    factory.assert_called_once_with(
        "dummy",
    )


def test_training_features_are_not_modified(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test that training features remain unchanged."""

    original = features.copy(
        deep=True
    )

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=DummyModel(),
        ):
            manager.train(
                features,
                target,
            )

    pd.testing.assert_frame_equal(
        features,
        original,
    )


def test_training_target_is_not_modified(
    manager: TrainingManager,
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """Test that training target remains unchanged."""

    original = target.copy(
        deep=True
    )

    with patch(
        "src.ml.training.training_manager.ModelSelector.resolve",
        return_value=("dummy",),
    ):
        with patch(
            "src.ml.training.training_manager.ModelFactory.create",
            return_value=DummyModel(),
        ):
            manager.train(
                features,
                target,
            )

    pd.testing.assert_series_equal(
        target,
        original,
    )


@pytest.mark.parametrize(
    "X,y,error,message",
    [
        (
            [1, 2, 3],
            pd.Series([1, 2, 3]),
            TypeError,
            "pandas DataFrame",
        ),
        (
            pd.DataFrame(
                {
                    "x": [1, 2, 3],
                }
            ),
            [1, 2, 3],
            TypeError,
            "pandas Series",
        ),
        (
            pd.DataFrame(),
            pd.Series([1]),
            ValueError,
            "cannot be empty",
        ),
        (
            pd.DataFrame(
                {
                    "x": [1],
                }
            ),
            pd.Series(),
            ValueError,
            "cannot be empty",
        ),
        (
            pd.DataFrame(
                {
                    "x": [1, 2],
                }
            ),
            pd.Series([1]),
            ValueError,
            "same number of rows",
        ),
    ],
)
def test_invalid_training_input_raises_error(
    manager: TrainingManager,
    X,
    y,
    error,
    message: str,
) -> None:
    """Test invalid training input."""

    with pytest.raises(
        error,
        match=message,
    ):
        manager.train(
            X,
            y,
        )