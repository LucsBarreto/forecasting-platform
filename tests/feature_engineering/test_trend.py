"""
Tests for trend feature engineering.
"""

import pandas as pd
import pytest

from src.config.pipeline import TrendFeatureSettings
from src.core.exceptions.validation import DataValidationError
from src.feature_engineering.trend import (
    TrendFeatureEngineer,
)


@pytest.fixture
def trend_config() -> TrendFeatureSettings:
    """Return a valid trend configuration."""

    return TrendFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
        ],
        group_levels=[
            ["COD CLIENTE"],
        ],
        periods=[
            1,
            2,
        ],
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample time series dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "COD CLIENTE": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
            "VOLUME": [
                10,
                20,
                30,
                100,
                200,
                300,
            ],
        }
    )


@pytest.fixture
def trend_engineer(
    trend_config: TrendFeatureSettings,
) -> TrendFeatureEngineer:
    """Return a trend feature engineer."""

    return TrendFeatureEngineer(
        config=trend_config,
    )


def test_create_trend_features(
    trend_engineer: TrendFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test trend feature creation."""

    result = trend_engineer.process(
        sample_dataframe,
    )

    assert (
        "VOLUME_trend_diff_1_COD CLIENTE"
        in result.columns
    )

    assert (
        "VOLUME_trend_growth_1_COD CLIENTE"
        in result.columns
    )

    assert (
        "VOLUME_trend_diff_2_COD CLIENTE"
        in result.columns
    )


def test_difference_is_calculated_correctly(
    trend_engineer: TrendFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test lagged difference calculation."""

    result = trend_engineer.process(
        sample_dataframe,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "A"
    ]

    values = client_a[
        "VOLUME_trend_diff_1_COD CLIENTE"
    ].tolist()

    assert pd.isna(values[0])
    assert values[1:] == [10, 10]


def test_growth_is_calculated_correctly(
    trend_engineer: TrendFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test percentage growth calculation."""

    result = trend_engineer.process(
        sample_dataframe,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "A"
    ]

    values = client_a[
        "VOLUME_trend_growth_1_COD CLIENTE"
    ].tolist()

    assert pd.isna(values[0])
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.5)


def test_trend_values_are_isolated_between_groups(
    trend_engineer: TrendFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Trend features must remain isolated between groups."""

    result = trend_engineer.process(
        sample_dataframe,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "A",
        "VOLUME_trend_diff_1_COD CLIENTE",
    ].tolist()

    client_b = result.loc[
        result["COD CLIENTE"] == "B",
        "VOLUME_trend_diff_1_COD CLIENTE",
    ].tolist()

    assert pd.isna(client_a[0])
    assert client_a[1:] == [10, 10]

    assert pd.isna(client_b[0])
    assert client_b[1:] == [100, 100]


def test_zero_previous_value_returns_nan_growth(
    trend_engineer: TrendFeatureEngineer,
) -> None:
    """Growth must be undefined when previous value is zero."""

    dataframe = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "COD CLIENTE": [
                "A",
                "A",
            ],
            "VOLUME": [
                0,
                100,
            ],
        }
    )

    result = trend_engineer.process(
        dataframe,
    )

    growth = result.loc[
        1,
        "VOLUME_trend_growth_1_COD CLIENTE",
    ]

    assert pd.isna(growth)


def test_disabled_features_return_original_dataframe(
    sample_dataframe: pd.DataFrame,
    trend_config: TrendFeatureSettings,
) -> None:
    """Disabled trend features must not modify data."""

    config = trend_config.model_copy(
        update={
            "enabled": False,
        }
    )

    engineer = TrendFeatureEngineer(
        config=config,
    )

    result = engineer.process(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_invalid_period_raises_error() -> None:
    """Invalid periods must raise an error."""

    config = TrendFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[["COD CLIENTE"]],
        periods=[0, 3],
    )

    with pytest.raises(DataValidationError):
        TrendFeatureEngineer._validate_configuration(
            config,
        )


def test_missing_dataframe_column_raises_error(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Missing columns must raise an error."""

    config = TrendFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[
            ["missing_column"],
        ],
        periods=[1],
    )

    with pytest.raises(DataValidationError):
        TrendFeatureEngineer._validate_dataframe(
            sample_dataframe,
            config,
        )