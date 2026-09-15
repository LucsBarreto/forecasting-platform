"""
Tests for lag feature engineering.
"""

import pandas as pd
import pytest

from src.core.exceptions import DataValidationError
from src.config.pipeline import LagFeatureSettings
from src.feature_engineering.lag import LagFeatureEngineer


@pytest.fixture
def lag_config() -> LagFeatureSettings:
    """Return a valid lag configuration."""

    return LagFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=[
            "VOLUME",
            "VALOR",
        ],
        group_levels=[
            ["COD CLIENTE"],
        ],
        lags=[
            1,
            2,
        ],
    )


@pytest.fixture
def lag_engineer(
    lag_config: LagFeatureSettings,
) -> LagFeatureEngineer:
    """Return a lag feature engineer instance."""

    return LagFeatureEngineer(
        config=lag_config,
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
            "COD ITEM": [
                "X",
                "X",
                "X",
                "X",
                "X",
                "X",
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



def test_create_lag_features(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Test creation of lag features."""

    data = lag_engineer._sort_dataframe(
        sample_dataframe,
        lag_config,
    )

    result = lag_engineer._create_lags(
        data,
        lag_config,
    )

    assert "VOLUME_lag_1_COD CLIENTE" in result.columns
    assert "VOLUME_lag_2_COD CLIENTE" in result.columns

    assert "VALOR_lag_1_COD CLIENTE" in result.columns
    assert "VALOR_lag_2_COD CLIENTE" in result.columns



def test_lag_values_are_isolated_between_groups(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Test that lag values remain isolated between groups."""

    data = lag_engineer._sort_dataframe(
        sample_dataframe,
        lag_config,
    )

    result = lag_engineer._create_lags(
        data,
        lag_config,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "A",
        "VOLUME_lag_1_COD CLIENTE",
    ].tolist()

    client_b = result.loc[
        result["COD CLIENTE"] == "B",
        "VOLUME_lag_1_COD CLIENTE",
    ].tolist()

    assert pd.isna(client_a[0])
    assert client_a[1:] == [10, 20]

    assert pd.isna(client_b[0])
    assert client_b[1:] == [100, 200]

def test_multiple_group_levels(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Test creation of lags for multiple grouping levels."""

    config = lag_config.model_copy(
        update={
            "group_levels": [
                ["COD CLIENTE"],
                ["COD ITEM"],
            ],
        }
    )

    data = lag_engineer._sort_dataframe(
        sample_dataframe,
        config,
    )

    result = lag_engineer._create_lags(
        data,
        config,
    )

    assert "VOLUME_lag_1_COD CLIENTE" in result.columns
    assert "VOLUME_lag_1_COD ITEM" in result.columns

    assert (
        "VOLUME_lag_1_COD CLIENTE"
        != "VOLUME_lag_1_COD ITEM"
    )

def test_invalid_lag_configuration() -> None:
    """Test invalid lag configuration."""

    config = LagFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[["COD CLIENTE"]],
        lags=[0, 1],
    )

    with pytest.raises(DataValidationError):
        LagFeatureEngineer._validate_configuration(
            config,
        )

def test_missing_dataframe_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test missing dataframe columns."""

    config = LagFeatureSettings(
        enabled=True,
        date_column="DATA",
        target_columns=["VOLUME"],
        group_levels=[
            ["COD CLIENTE", "coluna_inexistente"]
        ],
        lags=[1],
    )

    with pytest.raises(DataValidationError):
        LagFeatureEngineer._validate_dataframe(
            sample_dataframe,
            config,
        )

def test_disabled_lag_engineer_returns_original_dataframe(
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Disabled lag engineering must not modify the dataframe."""

    config = lag_config.model_copy(
        update={
            "enabled": False,
        }
    )

    engineer = LagFeatureEngineer(
        config=config,
    )

    result = engineer.process(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )

def test_process_creates_lag_features(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Process must create the configured lag features."""

    result = lag_engineer.process(
        sample_dataframe,
    )

    assert "VOLUME_lag_1_COD CLIENTE" in result.columns
    assert "VOLUME_lag_2_COD CLIENTE" in result.columns
    assert "VALOR_lag_1_COD CLIENTE" in result.columns
    assert "VALOR_lag_2_COD CLIENTE" in result.columns


def test_process_sorts_dataframe_chronologically(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Process must sort each group chronologically."""

    dataframe = sample_dataframe.sample(
        frac=1,
        random_state=42,
    )

    result = lag_engineer.process(
        dataframe,
    )

    for _, group in result.groupby(
        "COD CLIENTE",
        sort=False,
    ):
        assert group["DATA"].is_monotonic_increasing


def test_lag_preserves_original_columns(
    lag_engineer: LagFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Lag engineering must preserve original columns."""

    original_columns = set(
        sample_dataframe.columns,
    )

    result = lag_engineer.process(
        sample_dataframe,
    )

    assert original_columns.issubset(
        result.columns,
    )


def test_missing_date_column_raises_validation_error(
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Missing date column must raise DataValidationError."""

    dataframe = sample_dataframe.drop(
        columns=["DATA"],
    )

    engineer = LagFeatureEngineer(
        config=lag_config,
    )

    with pytest.raises(DataValidationError):
        engineer.process(dataframe)


def test_invalid_date_type_raises_validation_error(
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Non-datetime date column must raise DataValidationError."""

    dataframe = sample_dataframe.copy()

    dataframe["DATA"] = (
        dataframe["DATA"]
        .dt.strftime("%Y-%m-%d")
    )

    engineer = LagFeatureEngineer(
        config=lag_config,
    )

    with pytest.raises(DataValidationError):
        engineer.process(dataframe)



def test_null_group_values_are_preserved(
    sample_dataframe: pd.DataFrame,
    lag_config: LagFeatureSettings,
) -> None:
    """Null grouping values must remain isolated in their group."""

    dataframe = sample_dataframe.copy()

    dataframe.loc[
        dataframe["COD CLIENTE"] == "B",
        "COD CLIENTE",
    ] = None

    engineer = LagFeatureEngineer(
        config=lag_config,
    )

    result = engineer.process(
        dataframe,
    )

    null_group = result.loc[
        result["COD CLIENTE"].isna(),
        "VOLUME_lag_1_COD CLIENTE",
    ].tolist()

    assert pd.isna(null_group[0])
    assert null_group[1:] == [100, 200]



