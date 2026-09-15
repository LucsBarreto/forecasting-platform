"""
Integration tests for feature engineering pipeline.
"""

import pandas as pd
import pytest

from src.config.pipeline import (
    BusinessFeatureSettings,
    LagFeatureSettings,
    RollingFeatureSettings,
    TemporalFeatureSettings,
    TrendFeatureSettings,
)
from src.feature_engineering.business import BusinessFeatureEngineer
from src.feature_engineering.lag import LagFeatureEngineer
from src.feature_engineering.rolling import RollingFeatureEngineer
from src.feature_engineering.temporal import TemporalFeatureEngineer
from src.feature_engineering.trend import TrendFeatureEngineer


def create_dataframe() -> pd.DataFrame:
    """Create dataframe used in integration tests."""

    return pd.DataFrame(
        {
            "DATA": pd.date_range(
                "2025-01-01",
                periods=12,
                freq="D",
            ),
            "COD CLIENTE": ["C001"] * 12,
            "COD ITEM": ["P001"] * 12,
            "VOLUME": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
                110.0,
                120.0,
            ],
            "VALOR": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
            ],
        }
    )


def create_temporal_settings() -> TemporalFeatureSettings:
    """Create temporal feature settings."""

    return TemporalFeatureSettings(
        enabled=True,
        date_column="DATA",
        create_business_day=True,
        create_holiday=True,
        create_days_to_month_end=True,
        create_holiday_distance=True,
        create_cyclical_features=True,
    )


def create_lag_settings() -> LagFeatureSettings:
    """Create lag feature settings."""

    return LagFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE", "COD ITEM"],
        ],
        lags=[
            1,
            2,
        ],
    )


def create_rolling_settings() -> RollingFeatureSettings:
    """Create rolling feature settings."""

    return RollingFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE", "COD ITEM"],
        ],
        windows=[
            3,
            6,
        ],
    )


def create_trend_settings() -> TrendFeatureSettings:
    """Create trend feature settings."""

    return TrendFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE", "COD ITEM"],
        ],
    )


def create_business_settings() -> BusinessFeatureSettings:
    """Create business feature settings."""

    return BusinessFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE", "COD ITEM"],
        ],
    )


def apply_temporal_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply temporal feature engineering used by integration tests.

    TemporalFeatureEngineer currently depends on the configured
    holiday calendar and global settings, so the integration test
    uses its lower-level methods with explicit configuration.
    """

    config = create_temporal_settings()

    engineer = TemporalFeatureEngineer()

    result = engineer._create_calendar_features(
        data=data.copy(),
        date_column=config.date_column,
        holiday_dates=set(),
        config=config,
    )

    result = engineer._create_distance_features(
        data=result,
        date_column=config.date_column,
        holiday_dates=set(),
        config=config,
    )

    result = engineer._create_cyclical_features(
        data=result,
        date_column=config.date_column,
        config=config,
    )

    return result


def apply_all_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Apply all feature engineering modules sequentially."""

    result = apply_temporal_features(data)

    result = LagFeatureEngineer(
        config=create_lag_settings(),
    ).process(result)

    result = RollingFeatureEngineer(
        config=create_rolling_settings(),
    ).process(result)

    result = TrendFeatureEngineer(
        config=create_trend_settings(),
    ).process(result)

    result = BusinessFeatureEngineer(
        config=create_business_settings(),
    ).process(result)

    return result


def test_feature_engineering_modules_work_together() -> None:
    """
    Test that all feature engineering modules can run sequentially.
    """

    data = create_dataframe()

    original_columns = set(data.columns)

    result = apply_all_features(data)

    assert not result.empty

    assert original_columns.issubset(
        set(result.columns)
    )


def test_feature_engineering_creates_features_from_all_modules() -> None:
    """Test that each feature engineering module adds features."""

    data = create_dataframe()

    original_columns = set(data.columns)

    result = apply_temporal_features(data)

    assert set(result.columns) > original_columns

    columns_after_temporal = set(result.columns)

    result = LagFeatureEngineer(
        config=create_lag_settings(),
    ).process(result)

    assert set(result.columns) > columns_after_temporal

    columns_after_lag = set(result.columns)

    result = RollingFeatureEngineer(
        config=create_rolling_settings(),
    ).process(result)

    assert set(result.columns) > columns_after_lag

    columns_after_rolling = set(result.columns)

    result = TrendFeatureEngineer(
        config=create_trend_settings(),
    ).process(result)

    assert set(result.columns) > columns_after_rolling

    columns_after_trend = set(result.columns)

    result = BusinessFeatureEngineer(
        config=create_business_settings(),
    ).process(result)

    assert set(result.columns) > columns_after_trend


def test_feature_engineering_preserves_original_values() -> None:
    """Test that original target values are not modified."""

    data = create_dataframe()

    original_volume = data["VOLUME"].copy()
    original_valor = data["VALOR"].copy()

    result = apply_all_features(data)

    pd.testing.assert_series_equal(
        result["VOLUME"],
        original_volume,
        check_names=True,
    )

    pd.testing.assert_series_equal(
        result["VALOR"],
        original_valor,
        check_names=True,
    )


def test_feature_engineering_does_not_modify_input_dataframe() -> None:
    """Test that the original dataframe remains unchanged."""

    data = create_dataframe()

    original = data.copy(deep=True)

    apply_all_features(data)

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_feature_engineering_handles_multiple_groups() -> None:
    """Test that features remain isolated between groups."""

    data = pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C001",
                "C002",
                "C002",
            ],
            "COD ITEM": [
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "VOLUME": [
                10.0,
                20.0,
                100.0,
                200.0,
            ],
            "VALOR": [
                100.0,
                200.0,
                1000.0,
                2000.0,
            ],
        }
    )

    result = LagFeatureEngineer(
        config=create_lag_settings(),
    ).process(data)

    group_a = result[
        result["COD CLIENTE"] == "C001"
    ]

    group_b = result[
        result["COD CLIENTE"] == "C002"
    ]

    assert (
    group_a["VOLUME_lag_1_COD CLIENTE_COD ITEM"].iloc[-1]
    == 10.0
    )

    assert (
        group_b["VOLUME_lag_1_COD CLIENTE_COD ITEM"].iloc[-1]
        == 100.0
    )


def test_feature_engineering_output_has_same_row_count() -> None:
    """Feature engineering must not create or remove observations."""

    data = create_dataframe()

    original_row_count = len(data)

    result = apply_all_features(data)

    assert len(result) == original_row_count