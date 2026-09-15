"""
Tests for the data writer.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.writer import DataWriter


@pytest.fixture
def writer() -> DataWriter:
    """Return a data writer instance."""

    return DataWriter()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample dataframe."""

    return pd.DataFrame(
        {
            "cliente": ["A", "B", "C"],
            "VOLUME": [10, 20, 30],
            "VALOR": [100.0, 200.0, 300.0],
        }
    )


def test_write_csv(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test writing a dataframe to CSV."""

    destination = tmp_path / "output.csv"

    writer.write(
        sample_dataframe,
        destination,
    )

    assert destination.exists()


def test_write_parquet(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test writing a dataframe to Parquet."""

    destination = tmp_path / "output.parquet"

    writer.write(
        sample_dataframe,
        destination,
    )

    assert destination.exists()


def test_write_excel(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test writing a dataframe to Excel."""

    destination = tmp_path / "output.xlsx"

    writer.write(
        sample_dataframe,
        destination,
    )

    assert destination.exists()


def test_write_csv_preserves_DATA(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that CSV writing preserves dataframe data."""

    destination = tmp_path / "output.csv"

    writer.write(
        sample_dataframe,
        destination,
    )

    result = pd.read_csv(destination)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_write_parquet_preserves_DATA(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that Parquet writing preserves dataframe data."""

    destination = tmp_path / "output.parquet"

    writer.write(
        sample_dataframe,
        destination,
    )

    result = pd.read_parquet(destination)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )


def test_write_excel_preserves_DATA(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that Excel writing preserves dataframe data."""

    destination = tmp_path / "output.xlsx"

    writer.write(
        sample_dataframe,
        destination,
    )

    result = pd.read_excel(destination)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
        check_dtype=False,
    )


def test_write_is_case_insensitive(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that extensions are case insensitive."""

    destination = tmp_path / "output.CSV"

    writer.write(
        sample_dataframe,
        destination,
    )

    assert destination.exists()


def test_unsupported_extension_raises_error(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that unsupported extensions raise DataLoadingError."""

    destination = tmp_path / "output.json"

    with pytest.raises(
        DataLoadingError,
        match="formato de arquivo não suportado",
    ):
        writer.write(
            sample_dataframe,
            destination,
        )


def test_write_preserves_original_dataframe(
    writer: DataWriter,
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Test that writing does not modify the input dataframe."""

    original = sample_dataframe.copy(deep=True)

    writer.write(
        sample_dataframe,
        tmp_path / "output.csv",
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )