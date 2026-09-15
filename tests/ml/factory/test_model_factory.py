"""
Tests for ModelFactory.
"""

import pytest

from src.ml.factory import ModelFactory
from src.ml.models.base_model import BaseModel
from src.ml.models.baseline import BaselineModel
from src.ml.models.catboost import CatBoostModel
from src.ml.models.lightgbm import LightGBMModel
from src.ml.models.linear import LinearRegressionModel
from src.ml.models.prophet import ProphetModel
from src.ml.models.random_forest import RandomForestModel
from src.ml.models.sarima import SarimaModel
from src.ml.models.xgboost import XGBoostModel


def test_factory_creates_baseline_model() -> None:
    """Test baseline model creation."""

    model = ModelFactory.create(
        "baseline",
    )

    assert isinstance(
        model,
        BaselineModel,
    )

    assert isinstance(
        model,
        BaseModel,
    )


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        (
            "linear_regression",
            LinearRegressionModel,
        ),
        (
            "random_forest",
            RandomForestModel,
        ),
        (
            "xgboost",
            XGBoostModel,
        ),
        (
            "lightgbm",
            LightGBMModel,
        ),
        (
            "catboost",
            CatBoostModel,
        ),
        (
            "prophet",
            ProphetModel,
        ),
        (
            "sarima",
            SarimaModel,
        ),
    ],
)
def test_factory_creates_registered_models(
    name: str,
    expected_class: type[BaseModel],
) -> None:
    """Test creation of all registered models."""

    model = ModelFactory.create(name)

    assert isinstance(
        model,
        expected_class,
    )

    assert isinstance(
        model,
        BaseModel,
    )


def test_factory_accepts_uppercase_model_name() -> None:
    """Test model name normalization."""

    model = ModelFactory.create(
        "RANDOM_FOREST",
    )

    assert isinstance(
        model,
        RandomForestModel,
    )


def test_factory_accepts_model_name_with_whitespace() -> None:
    """Test whitespace normalization."""

    model = ModelFactory.create(
        "  random_forest  ",
    )

    assert isinstance(
        model,
        RandomForestModel,
    )


def test_factory_rejects_unknown_model() -> None:
    """Test unknown model handling."""

    with pytest.raises(
        ValueError,
        match="Unknown model",
    ):
        ModelFactory.create(
            "unknown_model",
        )


def test_factory_rejects_empty_model_name() -> None:
    """Test empty model name handling."""

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ModelFactory.create(
            "   ",
        )


def test_factory_rejects_non_string_model_name() -> None:
    """Test invalid model name type."""

    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        ModelFactory.create(  # type: ignore[arg-type]
            None,
        )


def test_available_models_returns_registered_models() -> None:
    """Test registered model listing."""

    models = ModelFactory.available_models()

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
def test_is_registered_returns_true(
    model_name: str,
) -> None:
    """Test registered model detection."""

    assert ModelFactory.is_registered(
        model_name,
    )


def test_is_registered_returns_false_for_unknown_model() -> None:
    """Test unknown model detection."""

    assert not ModelFactory.is_registered(
        "unknown_model",
    )


def test_is_registered_normalizes_model_name() -> None:
    """Test registration lookup normalization."""

    assert ModelFactory.is_registered(
        " RANDOM_FOREST ",
    )