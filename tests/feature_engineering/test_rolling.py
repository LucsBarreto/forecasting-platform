"""
Tests for rolling feature engineering.
"""

import pandas as pd
import pytest

from src.config.pipeline import RollingFeatureSettings
from src.core.exceptions.validation import DataValidationError
from src.feature_engineering.rolling import RollingFeatureEngineer


@pytest.fixture
def rolling_config() -> RollingFeatureSettings:
    """Return a valid rolling configuration."""

    return RollingFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE"],
        ],
        windows=[
            2,
            3,
        ],
        functions=[
            "mean",
            "std",
        ],
    )


@pytest.fixture
def rolling_engineer(
    rolling_config: RollingFeatureSettings,
) -> RollingFeatureEngineer:
    """Return a rolling feature engineer."""

    return RollingFeatureEngineer(
        config=rolling_config,
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return sample time series data."""

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
            "VALOR": [
                100,
                200,
                300,
                1000,
                2000,
                3000,
            ],
        }
    )


def test_create_rolling_features(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Test creation of rolling features."""

    data = rolling_engineer._sort_dataframe(
        sample_dataframe,
        rolling_config,
    )

    result = rolling_engineer._create_rolling_features(
        data,
        rolling_config,
    )

    assert (
        "VOLUME_rolling_mean_2_COD CLIENTE"
        in result.columns
    )

    assert (
        "VOLUME_rolling_std_2_COD CLIENTE"
        in result.columns
    )

    assert (
        "VALOR_rolling_mean_3_COD CLIENTE"
        in result.columns
    )


def test_rolling_does_not_use_current_value(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Rolling features must not contain the current target."""

    data = rolling_engineer._sort_dataframe(
        sample_dataframe,
        rolling_config,
    )

    result = rolling_engineer._create_rolling_features(
        data,
        rolling_config,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "A"
    ]

    assert pd.isna(
        client_a.iloc[0][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
    )

    assert (
        client_a.iloc[1][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
        == 10
    )

    assert (
        client_a.iloc[2][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
        == 15
    )


def test_rolling_isolated_between_groups(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Rolling features must remain isolated between groups."""

    data = rolling_engineer._sort_dataframe(
        sample_dataframe,
        rolling_config,
    )

    result = rolling_engineer._create_rolling_features(
        data,
        rolling_config,
    )

    client_b = result.loc[
        result["COD CLIENTE"] == "B"
    ]

    assert pd.isna(
        client_b.iloc[0][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
    )

    assert (
        client_b.iloc[1][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
        == 100
    )

    assert (
        client_b.iloc[2][
            "VOLUME_rolling_mean_2_COD CLIENTE"
        ]
        == 150
    )


def test_multiple_functions(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Test multiple rolling functions."""

    data = rolling_engineer._sort_dataframe(
        sample_dataframe,
        rolling_config,
    )

    result = rolling_engineer._create_rolling_features(
        data,
        rolling_config,
    )

    assert (
        "VOLUME_rolling_mean_2_COD CLIENTE"
        in result.columns
    )

    assert (
        "VOLUME_rolling_std_2_COD CLIENTE"
        in result.columns
    )


def test_multiple_group_levels(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Test rolling features for multiple group levels."""

    config = rolling_config.model_copy(
        update={
            "group_levels": [
                ["COD CLIENTE"],
                ["COD CLIENTE", "DATA"],
            ],
        }
    )

    data = rolling_engineer._sort_dataframe(
        sample_dataframe,
        config,
    )

    result = rolling_engineer._create_rolling_features(
        data,
        config,
    )

    assert (
        "VOLUME_rolling_mean_2_COD CLIENTE"
        in result.columns
    )

    assert (
        "VOLUME_rolling_mean_2_COD CLIENTE_DATA"
        in result.columns
    )


def test_disabled_configuration_returns_original_dataframe(
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Disabled rolling features must not modify the dataframe."""

    config = rolling_config.model_copy(
        update={"enabled": False}
    )

    engineer = RollingFeatureEngineer(
        config=config,
    )

    result = engineer.process(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_invalid_window_configuration() -> None:
    """Invalid rolling windows must raise an error."""

    config = RollingFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[["COD CLIENTE"]],
        windows=[0],
        functions=["mean"],
    )

    with pytest.raises(DataValidationError):
        RollingFeatureEngineer._validate_configuration(
            config,
        )


def test_duplicate_windows_configuration() -> None:
    """Duplicated rolling windows must raise an error."""

    config = RollingFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[["COD CLIENTE"]],
        windows=[3, 3],
        functions=["mean"],
    )

    with pytest.raises(DataValidationError):
        RollingFeatureEngineer._validate_configuration(
            config,
        )


def test_invalid_function_configuration() -> None:
    """Unsupported rolling functions must raise an error."""

    config = RollingFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[["COD CLIENTE"]],
        windows=[3],
        functions=["invalid"],
    )

    with pytest.raises(DataValidationError):
        RollingFeatureEngineer._validate_configuration(
            config,
        )


def test_missing_dataframe_columns(
    sample_dataframe: pd.DataFrame,
    rolling_config: RollingFeatureSettings,
) -> None:
    """Missing dataframe columns must raise an error."""

    dataframe = sample_dataframe.drop(
        columns=["VOLUME"]
    )

    with pytest.raises(DataValidationError):
        RollingFeatureEngineer._validate_dataframe(
            dataframe,
            rolling_config,
        )


def test_process_creates_features(
    rolling_engineer: RollingFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Process must execute the complete rolling pipeline."""

    result = rolling_engineer.process(
        sample_dataframe,
    )

    assert (
        "VOLUME_rolling_mean_2_COD CLIENTE"
        in result.columns
    )

    assert (
        "VALOR_rolling_mean_3_COD CLIENTE"
        in result.columns
    )