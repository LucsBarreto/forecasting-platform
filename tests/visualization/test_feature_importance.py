"""
Tests for FeatureImportanceVisualizer.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.core.exceptions.data import DataError
from src.visualization.feature_importance import (
    FeatureImportanceVisualizer,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return feature importance data."""

    return pd.DataFrame(
        {
            "feature": [
                "VOLUME",
                "VALOR",
                "clientes",
            ],
            "importance": [
                0.2,
                0.7,
                0.1,
            ],
        }
    )


def test_plot_returns_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Test feature importance chart."""

    visualizer = FeatureImportanceVisualizer()

    figure = visualizer.plot(
        dataframe,
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_plot_top_n(
    dataframe: pd.DataFrame,
) -> None:
    """Test top N filtering."""

    visualizer = FeatureImportanceVisualizer()

    figure = visualizer.plot(
        dataframe,
        top_n=2,
    )

    axis = figure.axes[0]

    assert len(
        axis.patches,
    ) == 2


def test_empty_dataframe_raises_error() -> None:
    """Test empty feature importance."""

    visualizer = FeatureImportanceVisualizer()

    with pytest.raises(
        DataError,
        match="empty feature importance",
    ):
        visualizer.plot(
            pd.DataFrame(),
        )


def test_missing_columns_raise_error() -> None:
    """Test required columns."""

    dataframe = pd.DataFrame(
        {
            "feature": [
                "VOLUME",
            ],
        }
    )

    visualizer = FeatureImportanceVisualizer()

    with pytest.raises(
        DataError,
        match="Missing columns",
    ):
        visualizer.plot(
            dataframe,
        )


def test_invalid_top_n_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test invalid top N."""

    visualizer = FeatureImportanceVisualizer()

    with pytest.raises(
        ValueError,
        match="top_n must be greater than zero",
    ):
        visualizer.plot(
            dataframe,
            top_n=0,
        )