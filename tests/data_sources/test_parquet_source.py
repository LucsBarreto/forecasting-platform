"""
Tests for Parquet data source.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.parquet_source import ParquetSource


@pytest.fixture
def source() -> ParquetSource:
    """Return a Parquet source instance."""

    return ParquetSource()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "COD CLIENTE": [1001, 1002, 1003],
            "COD ITEM": [10, 20, 30],
            "VOLUME": [100, 200, 300],
            "VALOR": [1000.0, 2500.0, 4000.0],
        }
    )


def test_read_returns_dataframe(
    source: ParquetSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that reading a Parquet file returns a dataframe."""

    source_path = Path("data.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        return_value=sample_dataframe,
    ) as mock_read:

        result = source.read(source_path)

    assert isinstance(result, pd.DataFrame)

    mock_read.assert_called_once()


def test_read_returns_expected_DATA(
    source: ParquetSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that Parquet data is returned correctly."""

    source_path = Path("data.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        return_value=sample_dataframe,
    ):

        result = source.read(source_path)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_read_passes_source_path_to_pandas(
    source: ParquetSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the source path is passed to pandas."""

    source_path = Path("data.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        return_value=sample_dataframe,
    ) as mock_read:

        source.read(source_path)

    args, _ = mock_read.call_args

    assert args[0] == source_path


def test_read_preserves_columns(
    source: ParquetSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that Parquet column names are preserved."""

    source_path = Path("data.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        return_value=sample_dataframe,
    ):

        result = source.read(source_path)

    assert list(result.columns) == list(
        sample_dataframe.columns
    )


def test_read_preserves_row_count(
    source: ParquetSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the number of rows is preserved."""

    source_path = Path("data.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        return_value=sample_dataframe,
    ):

        result = source.read(source_path)

    assert len(result) == len(sample_dataframe)


def test_read_raises_DATA_loading_error(
    source: ParquetSource,
) -> None:
    """Test that Parquet reading errors are converted."""

    source_path = Path("invalid.parquet")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        side_effect=ValueError("invalid parquet file"),
    ):

        with pytest.raises(DataLoadingError):
            source.read(source_path)


def test_read_preserves_original_exception(
    source: ParquetSource,
) -> None:
    """Test that the original exception is preserved as the cause."""

    source_path = Path("invalid.parquet")

    original_exception = ValueError("invalid parquet file")

    with patch(
        "src.data_sources.parquet_source.pd.read_parquet",
        side_effect=original_exception,
    ):

        with pytest.raises(DataLoadingError) as exc_info:
            source.read(source_path)

    assert exc_info.value.__cause__ is original_exception