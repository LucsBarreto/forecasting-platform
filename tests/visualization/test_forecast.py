"""
Tests for ForecastVisualizer.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.core.exceptions.data import DataError
from src.visualization.forecast import ForecastVisualizer


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return sample forecast data."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "actual": [
                30.0,
                10.0,
                20.0,
            ],
            "prediction": [
                28.0,
                11.0,
                19.0,
            ],
        }
    )


def test_plot_returns_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Test forecast chart."""

    visualizer = ForecastVisualizer()

    figure = visualizer.plot(
        dataframe,
        date_column="DATA",
        actual_column="actual",
        prediction_column="prediction",
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_plot_creates_two_lines(
    dataframe: pd.DataFrame,
) -> None:
    """Test actual and prediction lines."""

    visualizer = ForecastVisualizer()

    figure = visualizer.plot(
        dataframe,
        date_column="DATA",
        actual_column="actual",
        prediction_column="prediction",
    )

    axis = figure.axes[0]

    assert len(
        axis.lines,
    ) == 2


def test_plot_sorts_dates(
    dataframe: pd.DataFrame,
) -> None:
    """Test chronological sorting."""

    visualizer = ForecastVisualizer()

    figure = visualizer.plot(
        dataframe,
        date_column="DATA",
        actual_column="actual",
        prediction_column="prediction",
    )

    x_values = figure.axes[0].lines[0].get_xdata()

    assert list(x_values) == sorted(x_values)


def test_missing_column_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test missing columns."""

    visualizer = ForecastVisualizer()

    with pytest.raises(
        DataError,
        match="Missing columns",
    ):
        visualizer.plot(
            dataframe,
            date_column="DATA",
            actual_column="actual",
            prediction_column="missing",
        )


def test_invalid_dates_raise_error() -> None:
    """Test invalid dates."""

    dataframe = pd.DataFrame(
        {
            "DATA": [
                "invalid",
                "also-invalid",
            ],
            "actual": [
                10,
                20,
            ],
            "prediction": [
                11,
                19,
            ],
        }
    )

    visualizer = ForecastVisualizer()

    with pytest.raises(
        DataError,
        match="does not contain valid dates",
    ):
        visualizer.plot(
            dataframe,
            date_column="DATA",
            actual_column="actual",
            prediction_column="prediction",
        )