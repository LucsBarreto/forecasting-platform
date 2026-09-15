"""
Tests for the CSV data source.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.csv_source import CsvSource


@pytest.fixture
def source() -> CsvSource:
    """Return a CSV source instance."""

    return CsvSource()


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
    source: CsvSource,
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that reading a CSV returns a dataframe."""

    csv_path = tmp_path / "data.csv"

    sample_dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = source.read(csv_path)

    assert isinstance(result, pd.DataFrame)


def test_read_returns_expected_DATA(
    source: CsvSource,
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the CSV data is read correctly."""

    csv_path = tmp_path / "data.csv"

    sample_dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = source.read(csv_path)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_read_preserves_columns(
    source: CsvSource,
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that CSV column names are preserved."""

    csv_path = tmp_path / "data.csv"

    sample_dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = source.read(csv_path)

    assert list(result.columns) == list(
        sample_dataframe.columns
    )


def test_read_preserves_row_count(
    source: CsvSource,
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the number of rows is preserved."""

    csv_path = tmp_path / "data.csv"

    sample_dataframe.to_csv(
        csv_path,
        index=False,
    )

    result = source.read(csv_path)

    assert len(result) == len(sample_dataframe)


def test_read_empty_csv_returns_dataframe(
    source: CsvSource,
    tmp_path: Path,
) -> None:
    """Test reading a CSV containing only headers."""

    csv_path = tmp_path / "empty.csv"

    csv_path.write_text(
        "COD CLIENTE,COD ITEM,VOLUME,VALOR\n",
        encoding="utf-8",
    )

    result = source.read(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == [
        "COD CLIENTE",
        "COD ITEM",
        "VOLUME",
        "VALOR",
    ]


def test_read_missing_file_raises_DATA_loading_error(
    source: CsvSource,
    tmp_path: Path,
) -> None:
    """Test that a missing CSV raises DataLoadingError."""

    csv_path = tmp_path / "missing.csv"

    with pytest.raises(DataLoadingError):
        source.read(csv_path)


def test_read_invalid_csv_raises_DATA_loading_error(
    source: CsvSource,
    tmp_path: Path,
) -> None:
    """Test that an invalid CSV raises DataLoadingError."""

    csv_path = tmp_path / "invalid.csv"

    csv_path.write_bytes(b"\x80\x81\x82\x83")

    with pytest.raises(DataLoadingError):
        source.read(csv_path)


def test_read_accepts_path_object(
    source: CsvSource,
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that read accepts a pathlib.Path."""

    csv_path = tmp_path / "data.csv"

    sample_dataframe.to_csv(
        csv_path,
        index=False,
    )

    assert isinstance(csv_path, Path)

    result = source.read(csv_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_dataframe)