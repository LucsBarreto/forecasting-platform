"""
Tests for ModelMonitor.
"""

from datetime import datetime

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.monitoring import ModelMonitor


def test_monitor_starts_with_empty_history() -> None:
    """Test that monitor starts without records."""

    monitor = ModelMonitor()

    assert monitor.latest() is None
    assert monitor.history().empty


def test_record_returns_monitoring_record() -> None:
    """Test that record returns the created record."""

    monitor = ModelMonitor()

    result = monitor.record(
        model_name="random_forest",
        metrics={
            "mape": 10.5,
        },
        n_observations=100,
    )

    assert result["model_name"] == "random_forest"
    assert result["metrics"] == {
        "mape": 10.5,
    }
    assert result["n_observations"] == 100
    assert isinstance(
        result["timestamp"],
        datetime,
    )


def test_record_stores_execution() -> None:
    """Test that execution is stored in history."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="random_forest",
        metrics={
            "mape": 10.5,
        },
        n_observations=100,
    )

    assert len(
        monitor.history(),
    ) == 1


def test_history_returns_dataframe() -> None:
    """Test that history is returned as a DataFrame."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="linear_regression",
        metrics={
            "mape": 12.0,
        },
        n_observations=50,
    )

    result = monitor.history()

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert list(
        result.columns,
    ) == [
        "timestamp",
        "model_name",
        "metrics",
        "n_observations",
    ]


def test_history_contains_recorded_values() -> None:
    """Test that history contains recorded values."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="xgboost",
        metrics={
            "mape": 8.5,
            "mae": 15.0,
        },
        n_observations=200,
    )

    result = monitor.history()

    assert result.loc[0, "model_name"] == "xgboost"
    assert result.loc[0, "metrics"] == {
        "mape": 8.5,
        "mae": 15.0,
    }
    assert result.loc[0, "n_observations"] == 200


def test_latest_returns_latest_record() -> None:
    """Test that latest returns the most recent record."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="linear_regression",
        metrics={
            "mape": 15.0,
        },
        n_observations=100,
    )

    monitor.record(
        model_name="random_forest",
        metrics={
            "mape": 9.0,
        },
        n_observations=120,
    )

    result = monitor.latest()

    assert result is not None
    assert result["model_name"] == "random_forest"
    assert result["metrics"]["mape"] == 9.0


def test_multiple_records_are_preserved() -> None:
    """Test that multiple executions are preserved."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="linear_regression",
        metrics={
            "mape": 15.0,
        },
        n_observations=100,
    )

    monitor.record(
        model_name="random_forest",
        metrics={
            "mape": 10.0,
        },
        n_observations=100,
    )

    monitor.record(
        model_name="xgboost",
        metrics={
            "mape": 8.0,
        },
        n_observations=100,
    )

    assert len(
        monitor.history(),
    ) == 3


def test_clear_removes_history() -> None:
    """Test that monitoring history can be cleared."""

    monitor = ModelMonitor()

    monitor.record(
        model_name="random_forest",
        metrics={
            "mape": 10.0,
        },
        n_observations=100,
    )

    monitor.clear()

    assert monitor.latest() is None
    assert monitor.history().empty


def test_empty_model_name_raises_error() -> None:
    """Test that empty model names are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        DataValidationError,
        match="Model name cannot be empty",
    ):
        monitor.record(
            model_name="",
            metrics={
                "mape": 10.0,
            },
            n_observations=100,
        )


def test_invalid_model_name_type_raises_error() -> None:
    """Test that invalid model name types are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        TypeError,
        match="model_name must be a string",
    ):
        monitor.record(
            model_name=123,  # type: ignore[arg-type]
            metrics={
                "mape": 10.0,
            },
            n_observations=100,
        )


def test_empty_metrics_raise_error() -> None:
    """Test that empty metrics are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        DataValidationError,
        match="Metrics cannot be empty",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={},
            n_observations=100,
        )


def test_invalid_metrics_type_raises_error() -> None:
    """Test that invalid metrics types are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        TypeError,
        match="metrics must be a dictionary",
    ):
        monitor.record(
            model_name="random_forest",
            metrics=[],  # type: ignore[arg-type]
            n_observations=100,
        )


def test_non_numeric_metric_raises_error() -> None:
    """Test that non-numeric metrics are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={
                "mape": "10.0",  # type: ignore[dict-item]
            },
            n_observations=100,
        )


def test_null_metric_raises_error() -> None:
    """Test that null metrics are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        DataValidationError,
        match="Metrics cannot contain null values",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={
                "mape": float("nan"),
            },
            n_observations=100,
        )


def test_invalid_observation_type_raises_error() -> None:
    """Test that invalid observation counts are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        TypeError,
        match="n_observations must be an integer",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={
                "mape": 10.0,
            },
            n_observations=100.0,  # type: ignore[arg-type]
        )


def test_zero_observations_raise_error() -> None:
    """Test that zero observations are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        DataValidationError,
        match="greater than zero",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={
                "mape": 10.0,
            },
            n_observations=0,
        )


def test_negative_observations_raise_error() -> None:
    """Test that negative observations are rejected."""

    monitor = ModelMonitor()

    with pytest.raises(
        DataValidationError,
        match="greater than zero",
    ):
        monitor.record(
            model_name="random_forest",
            metrics={
                "mape": 10.0,
            },
            n_observations=-1,
        )


def test_record_does_not_modify_metrics() -> None:
    """Test that input metrics are not modified."""

    metrics = {
        "mape": 10.0,
        "mae": 20.0,
    }

    original = metrics.copy()

    monitor = ModelMonitor()

    monitor.record(
        model_name="random_forest",
        metrics=metrics,
        n_observations=100,
    )

    assert metrics == original