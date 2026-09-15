"""
Tests for model selection.
"""

from unittest.mock import patch

import pytest

from src.ml.training.model_selection import ModelSelector


def test_available_models_returns_registered_models() -> None:
    """Test registered model discovery."""

    selector = ModelSelector()

    models = selector.available_models()

    assert isinstance(
        models,
        tuple,
    )

    assert "linear_regression" in models
    assert "random_forest" in models
    assert "xgboost" in models
    assert "lightgbm" in models
    assert "catboost" in models
    assert "prophet" in models
    assert "sarima" in models


def test_enabled_models_returns_configured_models() -> None:
    """Test enabled model resolution."""

    selector = ModelSelector()

    models = selector.enabled_models()

    assert isinstance(
        models,
        tuple,
    )

    assert models

    for model in models:
        assert selector._is_enabled(
            __import__(
                "src.config",
                fromlist=["settings"],
            ).settings.models.models,
            model,
        )


def test_automl_returns_all_enabled_models() -> None:
    """Test AutoML model resolution."""

    selector = ModelSelector()

    with patch(
        "src.ml.training.model_selection.settings.models.automl.enabled",
        True,
    ):
        models = selector.resolve()

    assert models == selector.enabled_models()


def test_non_automl_returns_first_enabled_model() -> None:
    """Test single-model resolution."""

    selector = ModelSelector()

    with patch(
        "src.ml.training.model_selection.settings.models.automl.enabled",
        False,
    ):
        models = selector.resolve()

    assert len(models) == 1

    assert models[0] in selector.enabled_models()


def test_no_enabled_models_raises_error() -> None:
    """Test empty model configuration."""

    selector = ModelSelector()

    with patch(
        "src.ml.training.model_selection.ModelSelector.enabled_models",
        return_value=(),
    ):
        with pytest.raises(
            ValueError,
            match="No machine learning models are enabled",
        ):
            selector.resolve()


@pytest.mark.parametrize(
    "model_name",
    [
        "linear_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
        "prophet",
        "sarima",
    ],
)
def test_registered_models_can_be_checked(
    model_name: str,
) -> None:
    """Test individual model configuration lookup."""

    selector = ModelSelector()

    from src.config import settings

    result = selector._is_enabled(
        settings.models.models,
        model_name,
    )

    assert isinstance(
        result,
        bool,
    )