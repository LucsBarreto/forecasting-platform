"""
Tests for data aggregation.
"""

import pandas as pd
import pytest

from src.aggregation.aggregator import Aggregator
from src.config.pipeline import AggregationSettings
from src.core.exceptions.validation import DataValidationError

@pytest.fixture
def aggregation_config() -> AggregationSettings:
    """Return a valid aggregation configuration."""

    return AggregationSettings(
        group_by=[
            "DATA",
            "COD CLIENTE",
            "COD ITEM",
        ],
        metrics={
            "VOLUME": "sum",
            "VALOR": "sum",
        },
    )

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample transactional dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-02",
                ]
            ),
            "COD CLIENTE": [
                "A",
                "A",
                "B",
                "A",
                "B",
            ],
            "COD ITEM": [
                "X",
                "X",
                "X",
                "X",
                "Y",
            ],
            "VOLUME": [
                10,
                20,
                100,
                30,
                50,
            ],
            "VALOR": [
                100,
                200,
                1000,
                300,
                500,
            ],
        }
    )

@pytest.fixture
def aggregator() -> Aggregator:
    """Return an aggregator instance."""

    return Aggregator()

def test_aggregation_returns_expected_groups(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that aggregation creates the expected groups."""

    result = aggregator.aggregate(sample_dataframe)

    assert len(result) == 4

    assert set(
        [
            "DATA",
            "COD CLIENTE",
            "COD ITEM",
            "VOLUME",
            "VALOR",
        ]
    ).issubset(result.columns)

def test_aggregation_sums_volume_and_value(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that volume and value are summed correctly."""

    result = aggregator.aggregate(sample_dataframe)

    row = result[
        (result["DATA"] == pd.Timestamp("2025-01-01"))
        & (result["COD CLIENTE"] == "A")
        & (result["COD ITEM"] == "X")
    ].iloc[0]

    assert row["VOLUME"] == 30
    assert row["VALOR"] == 300

def test_aggregation_excludes_undeclared_metrics(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that undeclared price is not part of the result."""

    result = aggregator.aggregate(sample_dataframe)

    row = result[
        (result["DATA"] == pd.Timestamp("2025-01-01"))
        & (result["COD CLIENTE"] == "A")
        & (result["COD ITEM"] == "X")
    ].iloc[0]

    assert "preco" not in result.columns

def test_aggregation_excludes_undeclared_order_count(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that undeclared order count is not part of the result."""

    result = aggregator.aggregate(sample_dataframe)

    row = result[
        (result["DATA"] == pd.Timestamp("2025-01-01"))
        & (result["COD CLIENTE"] == "A")
        & (result["COD ITEM"] == "X")
    ].iloc[0]

    assert "pedidos" not in result.columns

def test_aggregation_preserves_original_dataframe(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that aggregation does not modify the input dataframe."""

    original = sample_dataframe.copy(deep=True)

    aggregator.aggregate(sample_dataframe)

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )

def test_aggregation_handles_multiple_groups(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that different customers and products remain isolated."""

    result = aggregator.aggregate(sample_dataframe)

    client_a = result[
        result["COD CLIENTE"] == "A"
    ]

    client_b = result[
        result["COD CLIENTE"] == "B"
    ]

    assert len(client_a) == 2
    assert len(client_b) == 2

def test_aggregation_handles_null_groups(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that null grouping values are preserved."""

    dataframe = sample_dataframe.copy()

    dataframe.loc[0, "COD CLIENTE"] = None

    result = aggregator.aggregate(dataframe)

    assert result["COD CLIENTE"].isna().any()

def test_missing_group_column_raises_error(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that missing grouping columns raise an error."""

    dataframe = sample_dataframe.drop(
        columns=["COD CLIENTE"]
    )

    with pytest.raises(DataValidationError):
        aggregator.aggregate(dataframe)

def test_missing_metric_column_raises_error(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
    ) -> None:
    """Test that missing metric columns raise an error."""

    dataframe = sample_dataframe.drop(
        columns=["VOLUME"]
    )

    with pytest.raises(DataValidationError):
        aggregator.aggregate(dataframe)

def test_validate_columns_accepts_valid_dataframe(
    sample_dataframe: pd.DataFrame,
    aggregation_config: AggregationSettings,
    ) -> None:
    """Test validation with a valid dataframe."""

    Aggregator._validate_columns(
        dataframe=sample_dataframe,
        config=aggregation_config,
    )

def test_validate_columns_rejects_missing_columns(
    sample_dataframe: pd.DataFrame,
    aggregation_config: AggregationSettings,
    ) -> None:
    """Test validation with missing columns."""

    dataframe = sample_dataframe.drop(
        columns=["VALOR"]
    )

    with pytest.raises(DataValidationError):
        Aggregator._validate_columns(
            dataframe=dataframe,
            config=aggregation_config,
        )

def test_aggregation_preserves_null_group_values(
    aggregator: Aggregator,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that null grouping values create a separate group."""

    dataframe = sample_dataframe.copy()

    dataframe.loc[0, "COD CLIENTE"] = None

    result = aggregator.aggregate(dataframe)

    null_groups = result[
        result["COD CLIENTE"].isna()
    ]

    assert len(null_groups) == 1

    row = null_groups.iloc[0]

    assert row["VOLUME"] == 10
    assert row["VALOR"] == 100


def test_validate_columns_rejects_all_missing_columns(
    aggregator: Aggregator,
    aggregation_config: AggregationSettings,
) -> None:
    """Test validation when all required columns are missing."""

    dataframe = pd.DataFrame(
        {
            "outra_coluna": [1, 2, 3],
        }
    )

    with pytest.raises(DataValidationError):
        Aggregator._validate_columns(
            dataframe=dataframe,
            config=aggregation_config,
        )