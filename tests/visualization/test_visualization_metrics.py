"""
Tests for MetricsVisualizer.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.core.exceptions.data import DataError
from src.visualization.metrics import MetricsVisualizer


def test_plot_returns_figure() -> None:
    """Test metric chart."""

    visualizer = MetricsVisualizer()

    figure = visualizer.plot(
        {
            "mae": 10.0,
            "rmse": 15.0,
            "mape": 0.1,
        }
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_plot_creates_expected_bars() -> None:
    """Test number of metric bars."""

    visualizer = MetricsVisualizer()

    figure = visualizer.plot(
        {
            "mae": 10.0,
            "rmse": 15.0,
        }
    )

    assert len(
        figure.axes[0].patches,
    ) == 2


def test_dataframe_plot_returns_figure() -> None:
    """Test dataframe metric chart."""

    dataframe = pd.DataFrame(
        {
            "metric": [
                "mae",
                "rmse",
            ],
            "value": [
                10.0,
                15.0,
            ],
        }
    )

    visualizer = MetricsVisualizer()

    figure = visualizer.plot_dataframe(
        dataframe,
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_empty_metrics_raise_error() -> None:
    """Test empty metric dictionary."""

    visualizer = MetricsVisualizer()

    with pytest.raises(
        DataError,
        match="empty metrics",
    ):
        visualizer.plot({})


def test_non_numeric_metric_raises_error() -> None:
    """Test invalid metric values."""

    visualizer = MetricsVisualizer()

    with pytest.raises(
        DataError,
        match="Metric values must be numeric",
    ):
        visualizer.plot(
            {
                "mae": "invalid",  # type: ignore[dict-item]
            }
        )


def test_missing_dataframe_columns_raise_error() -> None:
    """Test dataframe column validation."""

    dataframe = pd.DataFrame(
        {
            "metric": [
                "mae",
            ],
        }
    )

    visualizer = MetricsVisualizer()

    with pytest.raises(
        DataError,
        match="Missing columns",
    ):
        visualizer.plot_dataframe(
            dataframe,
        )