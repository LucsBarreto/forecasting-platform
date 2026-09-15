"""
Tests for business feature engineering.
"""

import pandas as pd
import pytest

from src.config.pipeline import BusinessFeatureSettings
from src.core.exceptions.validation import DataValidationError
from src.feature_engineering.business import BusinessFeatureEngineer


def create_settings(
    *,
    enabled: bool = True,
    date_column: str = "DATA",
    target_columns: list[str] | None = None,
    group_levels: list[list[str]] | None = None,
) -> BusinessFeatureSettings:
    """Create business feature settings for tests."""

    return BusinessFeatureSettings(
        enabled=enabled,
        date_column=date_column,
        target_columns=(
            target_columns
            if target_columns is not None
            else ["VOLUME", "VALOR"]
        ),
        group_levels=(
            group_levels
            if group_levels is not None
            else [
                ["COD CLIENTE"],
                ["COD CLIENTE", "COD ITEM"],
            ]
        ),
    )


def create_dataframe() -> pd.DataFrame:
    """Create dataframe for tests."""

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
                "C001",
                "C001",
                "C001",
                "C002",
                "C002",
                "C002",
            ],
            "COD ITEM": [
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
                "P001",
            ],
            "VOLUME": [
                10.0,
                20.0,
                30.0,
                100.0,
                200.0,
                300.0,
            ],
            "VALOR": [
                100.0,
                200.0,
                300.0,
                1000.0,
                2000.0,
                3000.0,
            ],
        }
    )


@pytest.fixture
def business_config() -> BusinessFeatureSettings:
    """Return a valid business feature configuration."""

    return create_settings()


@pytest.fixture
def business_engineer(
    business_config: BusinessFeatureSettings,
) -> BusinessFeatureEngineer:
    """Return a business feature engineer."""

    return BusinessFeatureEngineer(
        config=business_config,
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return create_dataframe()


def test_process_creates_business_features(
    business_engineer: BusinessFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that process creates business features."""

    result = business_engineer.process(
        sample_dataframe,
    )

    assert "VOLUME_growth_COD CLIENTE" in result.columns
    assert "VALOR_growth_COD CLIENTE" in result.columns

    assert "VOLUME_share_COD CLIENTE" in result.columns
    assert "VALOR_share_COD CLIENTE" in result.columns


def test_process_preserves_original_columns(
    business_engineer: BusinessFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that original columns are preserved."""

    original_columns = set(
        sample_dataframe.columns
    )

    result = business_engineer.process(
        sample_dataframe,
    )

    assert original_columns.issubset(
        set(result.columns)
    )


def test_process_does_not_modify_input_dataframe(
    business_engineer: BusinessFeatureEngineer,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that processing does not modify the input dataframe."""

    original = sample_dataframe.copy(
        deep=True
    )

    business_engineer.process(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_disabled_business_features_return_original_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that disabled business features return the input unchanged."""

    config = create_settings(
        enabled=False,
    )

    engineer = BusinessFeatureEngineer(
        config=config,
    )

    result = engineer.process(
        sample_dataframe,
    )

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_validate_configuration_accepts_valid_configuration(
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that a valid configuration is accepted."""

    BusinessFeatureEngineer._validate_configuration(
        business_config,
    )


def test_validate_configuration_rejects_empty_targets() -> None:
    """Test that empty target configuration raises an error."""

    config = create_settings(
        target_columns=[],
    )

    with pytest.raises(DataValidationError):
        BusinessFeatureEngineer._validate_configuration(
            config,
        )


def test_validate_configuration_rejects_empty_groups() -> None:
    """Test that empty group configuration raises an error."""

    config = create_settings(
        group_levels=[],
    )

    with pytest.raises(DataValidationError):
        BusinessFeatureEngineer._validate_configuration(
            config,
        )


def test_validate_dataframe_accepts_valid_dataframe(
    business_config: BusinessFeatureSettings,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test dataframe validation with valid columns."""

    BusinessFeatureEngineer._validate_dataframe(
        sample_dataframe,
        business_config,
    )


def test_validate_dataframe_rejects_missing_target() -> None:
    """Test that missing target columns raise an error."""

    dataframe = create_dataframe().drop(
        columns=["VOLUME"],
    )

    config = create_settings()

    with pytest.raises(DataValidationError):
        BusinessFeatureEngineer._validate_dataframe(
            dataframe,
            config,
        )


def test_validate_dataframe_rejects_missing_date() -> None:
    """Test that missing date column raises an error."""

    dataframe = create_dataframe().drop(
        columns=["DATA"],
    )

    config = create_settings()

    with pytest.raises(DataValidationError):
        BusinessFeatureEngineer._validate_dataframe(
            dataframe,
            config,
        )


def test_validate_dataframe_rejects_missing_group_column() -> None:
    """Test that missing grouping columns raise an error."""

    dataframe = create_dataframe().drop(
        columns=["COD CLIENTE"],
    )

    config = create_settings()

    with pytest.raises(DataValidationError):
        BusinessFeatureEngineer._validate_dataframe(
            dataframe,
            config,
        )


def test_growth_features_are_created(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that growth features are created for each target."""

    result = BusinessFeatureEngineer._create_growth_features(
        sample_dataframe,
        business_config,
    )

    expected_columns = {
        "VOLUME_growth_COD CLIENTE",
        "VALOR_growth_COD CLIENTE",
        "VOLUME_growth_COD CLIENTE_COD ITEM",
        "VALOR_growth_COD CLIENTE_COD ITEM",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_growth_features_use_previous_value(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that growth uses the previous observation."""

    result = BusinessFeatureEngineer._create_growth_features(
        sample_dataframe,
        business_config,
    )

    growth = result[
        "VOLUME_growth_COD CLIENTE"
    ]

    assert pd.isna(growth.iloc[0])

    assert growth.iloc[1] == pytest.approx(1.0)

    assert growth.iloc[2] == pytest.approx(0.5)


def test_growth_features_are_isolated_between_groups(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that growth features do not mix different groups."""

    result = BusinessFeatureEngineer._create_growth_features(
        sample_dataframe,
        business_config,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "C001",
        "VOLUME_growth_COD CLIENTE",
    ].tolist()

    client_b = result.loc[
        result["COD CLIENTE"] == "C002",
        "VOLUME_growth_COD CLIENTE",
    ].tolist()

    assert pd.isna(client_a[0])
    assert client_a[1:] == [
        pytest.approx(1.0),
        pytest.approx(0.5),
    ]

    assert pd.isna(client_b[0])
    assert client_b[1:] == [
        pytest.approx(1.0),
        pytest.approx(0.5),
    ]


def test_growth_features_support_multiple_targets(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that growth features are created for all targets."""

    result = BusinessFeatureEngineer._create_growth_features(
        sample_dataframe,
        business_config,
    )

    for target in business_config.target_columns:
        for group_level in business_config.group_levels:

            suffix = "_".join(group_level)

            feature = (
                f"{target}_growth_{suffix}"
            )

            assert feature in result.columns


def test_share_features_are_created(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that share features are created."""

    result = BusinessFeatureEngineer._create_share_features(
        sample_dataframe,
        business_config,
    )

    expected_columns = {
        "VOLUME_share_COD CLIENTE",
        "VALOR_share_COD CLIENTE",
        "VOLUME_share_COD CLIENTE_COD ITEM",
        "VALOR_share_COD CLIENTE_COD ITEM",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_share_features_sum_to_one_within_group(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that shares sum to one within each group."""

    result = BusinessFeatureEngineer._create_share_features(
        sample_dataframe,
        business_config,
    )

    shares = (
        result.groupby(
            ["COD CLIENTE"],
            sort=False,
        )["VOLUME_share_COD CLIENTE"]
        .sum()
    )

    assert shares.loc["C001"] == pytest.approx(1.0)
    assert shares.loc["C002"] == pytest.approx(1.0)


def test_share_features_are_isolated_between_groups(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that shares do not mix different groups."""

    result = BusinessFeatureEngineer._create_share_features(
        sample_dataframe,
        business_config,
    )

    client_a = result.loc[
        result["COD CLIENTE"] == "C001",
        "VOLUME_share_COD CLIENTE",
    ]

    client_b = result.loc[
        result["COD CLIENTE"] == "C002",
        "VOLUME_share_COD CLIENTE",
    ]

    assert client_a.tolist() == [
        pytest.approx(1 / 6),
        pytest.approx(2 / 6),
        pytest.approx(3 / 6),
    ]

    assert client_b.tolist() == [
        pytest.approx(1 / 6),
        pytest.approx(2 / 6),
        pytest.approx(3 / 6),
    ]


def test_share_features_support_multiple_targets(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that share features are created for all targets."""

    result = BusinessFeatureEngineer._create_share_features(
        sample_dataframe,
        business_config,
    )

    for target in business_config.target_columns:
        for group_level in business_config.group_levels:

            suffix = "_".join(group_level)

            feature = (
                f"{target}_share_{suffix}"
            )

            assert feature in result.columns


def test_create_growth_features_preserves_input(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that growth feature creation does not modify input."""

    original = sample_dataframe.copy(
        deep=True
    )

    BusinessFeatureEngineer._create_growth_features(
        sample_dataframe,
        business_config,
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_create_share_features_preserves_input(
    sample_dataframe: pd.DataFrame,
    business_config: BusinessFeatureSettings,
) -> None:
    """Test that share feature creation does not modify input."""

    original = sample_dataframe.copy(
        deep=True
    )

    BusinessFeatureEngineer._create_share_features(
        sample_dataframe,
        business_config,
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )