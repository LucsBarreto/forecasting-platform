"""
Tests for ModelRegistry.
"""

import pytest

from src.ml.models.base_model import BaseModel
from src.ml.models.random_forest import RandomForestModel
from src.ml.registry import ModelRegistry


def test_get_registered_model() -> None:
    """Test retrieving a registered model."""

    model_class = ModelRegistry.get(
        "random_forest",
    )

    assert model_class is RandomForestModel


def test_get_registered_model_is_base_model() -> None:
    """Test that registered models inherit from BaseModel."""

    model_class = ModelRegistry.get(
        "random_forest",
    )

    assert issubclass(
        model_class,
        BaseModel,
    )


def test_get_normalizes_model_name() -> None:
    """Test model name normalization."""

    model_class = ModelRegistry.get(
        " RANDOM_FOREST ",
    )

    assert model_class is RandomForestModel


def test_get_unknown_model_raises_error() -> None:
    """Test unknown model handling."""

    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        ModelRegistry.get(
            "unknown_model",
        )


def test_available_models() -> None:
    """Test registered model listing."""

    models = ModelRegistry.available_models()

    assert isinstance(
        models,
        tuple,
    )

    assert set(models) == {
        "baseline",
        "linear_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "prophet",
        "sarima",
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "baseline",
        "linear_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "prophet",
        "sarima",
    ],
)
def test_is_registered(
    model_name: str,
) -> None:
    """Test registered model detection."""

    assert ModelRegistry.is_registered(
        model_name,
    )


def test_unknown_model_is_not_registered() -> None:
    """Test unknown model detection."""

    assert not ModelRegistry.is_registered(
        "unknown_model",
    )


def test_register_custom_model() -> None:
    """Test registering a new model."""

    class CustomModel(BaseModel):

        @property
        def name(self) -> str:
            return "custom_model"

        def fit(
            self,
            X,
            y,
        ):
            return self

        def predict(
            self,
            X,
        ):
            return X.iloc[:, 0]

    ModelRegistry.register(
        "custom_model",
        CustomModel,
    )

    assert ModelRegistry.is_registered(
        "custom_model",
    )

    assert (
        ModelRegistry.get("custom_model")
        is CustomModel
    )


def test_register_rejects_invalid_model_class() -> None:
    """Test invalid model class handling."""

    class InvalidModel:
        pass

    with pytest.raises(
        TypeError,
        match="must inherit from BaseModel",
    ):
        ModelRegistry.register(
            "invalid_model",
            InvalidModel,  # type: ignore[arg-type]
        )


def test_register_rejects_duplicate_model() -> None:
    """Test duplicate model registration."""

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        ModelRegistry.register(
            "random_forest",
            RandomForestModel,
        )


def test_register_rejects_empty_name() -> None:
    """Test empty registration name."""

    class CustomModel(BaseModel):

        @property
        def name(self) -> str:
            return "custom_model"

        def fit(
            self,
            X,
            y,
        ):
            return self

        def predict(
            self,
            X,
        ):
            return X.iloc[:, 0]

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ModelRegistry.register(
            "   ",
            CustomModel,
        )


def test_is_registered_rejects_non_string_name() -> None:
    """Test invalid model name type."""

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        ModelRegistry.is_registered(  # type: ignore[arg-type]
            None,
        )