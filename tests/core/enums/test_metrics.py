"""
Tests for MetricType enum.
"""

from src.core.enums.metrics import MetricType


def test_metric_type_values() -> None:
    """Test supported metric values."""

    assert MetricType.MAE == "mae"
    assert MetricType.RMSE == "rmse"
    assert MetricType.MAPE == "mape"
    assert MetricType.SMAPE == "smape"
    assert MetricType.R2 == "r2"


def test_metric_type_is_str_enum() -> None:
    """Test that MetricType members behave as strings."""

    assert isinstance(MetricType.MAE, str)


def test_metric_type_contains_expected_members() -> None:
    """Test all expected members exist."""

    assert set(MetricType) == {
        MetricType.MAE,
        MetricType.RMSE,
        MetricType.MAPE,
        MetricType.SMAPE,
        MetricType.R2,
    }