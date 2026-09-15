"""
Tests for EDAVisualizer.
"""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.core.exceptions.data import DataError
from src.visualization.eda import EDAVisualizer


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return sample dataframe."""

    return pd.DataFrame(
        {
            "VOLUME": [10, 20, None, 40],
            "VALOR": [100, 200, 300, 400],
        }
    )


def test_missing_values_returns_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Test missing values chart."""

    visualizer = EDAVisualizer()

    figure = visualizer.plot_missing_values(
        dataframe,
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_numeric_distribution_returns_figure(
    dataframe: pd.DataFrame,
) -> None:
    """Test numeric distribution chart."""

    visualizer = EDAVisualizer()

    figure = visualizer.plot_numeric_distribution(
        dataframe,
        column="VALOR",
    )

    assert isinstance(
        figure,
        Figure,
    )


def test_missing_column_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test missing column."""

    visualizer = EDAVisualizer()

    with pytest.raises(
        DataError,
        match="Column 'missing' was not found",
    ):
        visualizer.plot_numeric_distribution(
            dataframe,
            column="missing",
        )


def test_non_numeric_column_raises_error() -> None:
    """Test non-numeric distribution."""

    dataframe = pd.DataFrame(
        {
            "categoria": [
                "A",
                "B",
            ],
        }
    )

    visualizer = EDAVisualizer()

    with pytest.raises(
        DataError,
        match="must be numeric",
    ):
        visualizer.plot_numeric_distribution(
            dataframe,
            column="categoria",
        )


def test_empty_dataframe_raises_error() -> None:
    """Test empty dataframe."""

    visualizer = EDAVisualizer()

    with pytest.raises(
        DataError,
        match="empty dataframe",
    ):
        visualizer.plot_missing_values(
            pd.DataFrame(),
        )


def test_invalid_bins_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test invalid histogram bins."""

    visualizer = EDAVisualizer()

    with pytest.raises(
        ValueError,
        match="bins must be greater than zero",
    ):
        visualizer.plot_numeric_distribution(
            dataframe,
            column="VALOR",
            bins=0,
        )