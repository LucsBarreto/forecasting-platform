"""
Tests for ModelType enum.
"""

from src.core.enums.model import ModelType


def test_model_type_values() -> None:
    """Test supported model values."""

    assert ModelType.LINEAR_REGRESSION == "linear_regression"
    assert ModelType.RANDOM_FOREST == "random_forest"
    assert ModelType.XGBOOST == "xgboost"
    assert ModelType.LIGHTGBM == "lightgbm"
    assert ModelType.CATBOOST == "catboost"
    assert ModelType.PROPHET == "prophet"
    assert ModelType.SARIMA == "sarima"


def test_model_type_is_str_enum() -> None:
    """Test that ModelType members behave as strings."""

    assert isinstance(ModelType.XGBOOST, str)


def test_model_type_contains_expected_members() -> None:
    """Test all expected members exist."""

    assert set(ModelType) == {
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.CATBOOST,
        ModelType.PROPHET,
        ModelType.SARIMA,
    }