"""
Tests for FeatureSelector.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.feature_selection import FeatureSelector


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "COD CLIENTE": [
                "C001",
                "C002",
                "C003",
            ],
            "COD ITEM": [
                "P001",
                "P002",
                "P003",
            ],
            "VOLUME": [
                10.0,
                20.0,
                30.0,
            ],
            "VALOR": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )


def test_select_returns_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Test that selection returns a dataframe."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "VALOR",
        ],
    )

    result = selector.select(
        dataframe,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_select_returns_only_configured_features(
    dataframe: pd.DataFrame,
) -> None:
    """Test that only configured features are returned."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "VALOR",
        ],
    )

    result = selector.select(
        dataframe,
    )

    assert list(
        result.columns,
    ) == [
        "VOLUME",
        "VALOR",
    ]


def test_select_preserves_feature_order(
    dataframe: pd.DataFrame,
) -> None:
    """Test that configured feature order is preserved."""

    selector = FeatureSelector(
        features=[
            "VALOR",
            "VOLUME",
            "COD CLIENTE",
        ],
    )

    result = selector.select(
        dataframe,
    )

    assert list(
        result.columns,
    ) == [
        "VALOR",
        "VOLUME",
        "COD CLIENTE",
    ]


def test_select_preserves_dataframe_index(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the input index is preserved."""

    dataframe.index = [
        10,
        20,
        30,
    ]

    selector = FeatureSelector(
        features=[
            "VOLUME",
        ],
    )

    result = selector.select(
        dataframe,
    )

    pd.testing.assert_index_equal(
        result.index,
        dataframe.index,
    )


def test_select_preserves_feature_values(
    dataframe: pd.DataFrame,
) -> None:
    """Test that feature values are preserved."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "VALOR",
        ],
    )

    result = selector.select(
        dataframe,
    )

    expected = dataframe[
        [
            "VOLUME",
            "VALOR",
        ]
    ]

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_select_does_not_modify_input(
    dataframe: pd.DataFrame,
) -> None:
    """Test that the input dataframe is not modified."""

    original = dataframe.copy(
        deep=True,
    )

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "VALOR",
        ],
    )

    selector.select(
        dataframe,
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )




def test_duplicated_features_raise_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test that duplicated features are rejected."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "VOLUME",
        ],
    )

    with pytest.raises(
        DataValidationError,
        match="Duplicated features are not allowed",
    ):
        selector.select(
            dataframe,
        )


def test_empty_feature_name_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test that empty feature names are rejected."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "",
        ],
    )

    with pytest.raises(
        DataValidationError,
        match="Feature names must be non-empty strings",
    ):
        selector.select(
            dataframe,
        )


def test_missing_feature_raises_error(
    dataframe: pd.DataFrame,
) -> None:
    """Test that missing features are rejected."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
            "missing_feature",
        ],
    )

    with pytest.raises(
        DataValidationError,
        match="Missing features: missing_feature",
    ):
        selector.select(
            dataframe,
        )


def test_invalid_dataframe_type_raises_error() -> None:
    """Test that invalid dataframe types are rejected."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
        ],
    )

    with pytest.raises(
        TypeError,
        match="dataframe must be a pandas DataFrame",
    ):
        selector.select(
            [1, 2, 3],  # type: ignore[arg-type]
        )


def test_result_is_independent_from_input(
    dataframe: pd.DataFrame,
) -> None:
    """Test that result does not share mutable data with input."""

    selector = FeatureSelector(
        features=[
            "VOLUME",
        ],
    )

    result = selector.select(
        dataframe,
    )

    result.loc[0, "VOLUME"] = 999.0

    assert dataframe.loc[0, "VOLUME"] == 10.0
    
def test_selects_all_features_when_no_features_are_configured():
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
        }
    )

    selector = FeatureSelector(features=[])

    result = selector.select(dataframe)

    pd.testing.assert_frame_equal(
        result,
        dataframe,
    )