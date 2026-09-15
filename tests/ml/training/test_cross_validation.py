"""
Tests for temporal data splitting.
"""

import pandas as pd
import pytest

from src.core.exceptions.validation import DataValidationError
from src.ml.training.cross_validation import (
    TemporalSplit,
    TemporalSplitter,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """Return a chronological dataframe."""

    return pd.DataFrame(
        {
            "DATA": pd.date_range(
                "2020-01-01",
                periods=20,
                freq="D",
            ),
            "feature": range(20),
            "target": range(100, 120),
        }
    )


@pytest.fixture
def splitter() -> TemporalSplitter:
    """Return a default temporal splitter."""

    return TemporalSplitter(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
    )


def test_split_returns_temporal_split(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test split result type."""

    result = splitter.split(dataframe)

    assert isinstance(
        result,
        TemporalSplit,
    )


def test_split_returns_expected_sizes(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test train, validation and test sizes."""

    result = splitter.split(dataframe)

    assert len(result.train) == 14
    assert len(result.validation) == 3
    assert len(result.test) == 3


def test_split_preserves_temporal_order(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test chronological ordering."""

    result = splitter.split(dataframe)

    assert result.train["DATA"].is_monotonic_increasing
    assert result.validation["DATA"].is_monotonic_increasing
    assert result.test["DATA"].is_monotonic_increasing


def test_split_has_no_temporal_overlap(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test that splits do not overlap."""

    result = splitter.split(dataframe)

    assert (
        result.train["DATA"].max()
        < result.validation["DATA"].min()
    )

    assert (
        result.validation["DATA"].max()
        < result.test["DATA"].min()
    )


def test_split_sorts_unsorted_dataframe(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test that input data is sorted chronologically."""

    shuffled = dataframe.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    result = splitter.split(shuffled)

    assert result.train["DATA"].is_monotonic_increasing
    assert result.validation["DATA"].is_monotonic_increasing
    assert result.test["DATA"].is_monotonic_increasing


def test_split_does_not_modify_input(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test that the input dataframe remains unchanged."""

    original = dataframe.copy(deep=True)

    splitter.split(dataframe)

    pd.testing.assert_frame_equal(
        dataframe,
        original,
    )


def test_missing_date_column_raises_error(
    dataframe: pd.DataFrame,
    splitter: TemporalSplitter,
) -> None:
    """Test missing date column."""

    dataframe = dataframe.drop(
        columns=["DATA"]
    )

    with pytest.raises(
        DataValidationError,
        match="Date column 'DATA' was not found",
    ):
        splitter.split(dataframe)


def test_empty_dataframe_raises_error(
    splitter: TemporalSplitter,
) -> None:
    """Test empty dataframe."""

    dataframe = pd.DataFrame(
        columns=[
            "DATA",
            "feature",
        ]
    )

    with pytest.raises(
        DataValidationError,
        match="empty dataframe",
    ):
        splitter.split(dataframe)


def test_invalid_dates_raise_error(
    splitter: TemporalSplitter,
) -> None:
    """Test invalid dates."""

    dataframe = pd.DataFrame(
        {
            "DATA": [
                "2020-01-01",
                "invalid-date",
                "2020-01-03",
            ],
            "feature": [1, 2, 3],
        }
    )

    with pytest.raises(
        DataValidationError,
        match="contains invalid dates",
    ):
        splitter.split(dataframe)


@pytest.mark.parametrize(
    "train,validation,test",
    [
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (-0.1, 0.6, 0.5),
        (0.6, -0.1, 0.5),
        (0.6, 0.5, -0.1),
    ],
)
def test_invalid_split_sizes_raise_error(
    train: float,
    validation: float,
    test: float,
) -> None:
    """Test invalid split proportions."""

    splitter = TemporalSplitter(
        train_size=train,
        validation_size=validation,
        test_size=test,
    )

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        splitter._validate_configuration()


def test_split_sizes_must_sum_to_one() -> None:
    """Test invalid split proportion sum."""

    splitter = TemporalSplitter(
        train_size=0.70,
        validation_size=0.20,
        test_size=0.20,
    )

    with pytest.raises(
        ValueError,
        match="must sum to 1",
    ):
        splitter._validate_configuration()


def test_non_dataframe_raises_error(
    splitter: TemporalSplitter,
) -> None:
    """Test invalid dataframe type."""

    with pytest.raises(
        TypeError,
        match="pandas DataFrame",
    ):
        splitter.split([1, 2, 3])  # type: ignore[arg-type]


def test_custom_date_column() -> None:
    """Test custom date column."""

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2020-01-01",
                periods=20,
                freq="D",
            ),
            "value": range(20),
        }
    )

    splitter = TemporalSplitter(
        train_size=0.70,
        validation_size=0.15,
        test_size=0.15,
        date_column="date",
    )

    result = splitter.split(dataframe)

    assert len(result.train) == 14
    assert len(result.validation) == 3
    assert len(result.test) == 3