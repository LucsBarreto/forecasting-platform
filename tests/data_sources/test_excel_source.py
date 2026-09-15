"""
Tests for Excel data source.
"""

from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.excel_source import ExcelSource, resolve_excel_engine


@pytest.fixture
def source() -> ExcelSource:
    """Return an Excel source instance."""

    return ExcelSource()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
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
            "COD CLIENTE": [1001, 1002, 1003],
            "COD ITEM": [101, 102, 103],
            "VOLUME": [10, 20, 30],
            "VALOR": [100.0, 200.0, 300.0],
        }
    )


def test_read_returns_dataframe(
    source: ExcelSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that Excel source returns a dataframe."""

    source_path = Path("test.xlsx")

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        return_value=sample_dataframe,
    ) as mock_read:

        result = source.read(source_path)

    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(
        result,
        sample_dataframe,
    )

    mock_read.assert_called_once()


def test_read_uses_configured_sheet_name(
    source: ExcelSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the configured sheet name is passed to pandas."""

    from src.config import settings

    source_path = Path("test.xlsx")

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        return_value=sample_dataframe,
    ) as mock_read:

        source.read(source_path)

    _, kwargs = mock_read.call_args

    assert kwargs["sheet_name"] == settings.data.sheet_name


def test_read_passes_source_path_to_pandas(
    source: ExcelSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the source path is passed to pandas."""

    source_path = Path("test.xlsx")

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        return_value=sample_dataframe,
    ) as mock_read:

        source.read(source_path)

    args, _ = mock_read.call_args

    assert args[0] == source_path


def test_read_preserves_dataframe(
    source: ExcelSource,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Test that the dataframe returned by pandas is preserved."""

    source_path = Path("test.xlsx")

    original = sample_dataframe.copy(deep=True)

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        return_value=sample_dataframe,
    ):

        result = source.read(source_path)

    pd.testing.assert_frame_equal(
        result,
        original,
    )


def test_read_raises_DATA_loading_error(
    source: ExcelSource,
) -> None:
    """Test that Excel reading errors are converted."""

    source_path = Path("invalid.xlsx")

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        side_effect=ValueError("invalid excel file"),
    ):

        with pytest.raises(DataLoadingError):
            source.read(source_path)


def test_read_preserves_original_exception(
    source: ExcelSource,
) -> None:
    """Test that the original exception is preserved as the cause."""

    source_path = Path("invalid.xlsx")

    original_exception = ValueError("invalid excel file")

    with patch(
        "src.data_sources.excel_source.pd.read_excel",
        side_effect=original_exception,
    ):

        with pytest.raises(DataLoadingError) as exc_info:
            source.read(source_path)

    assert exc_info.value.__cause__ is original_exception


def test_xlsb_extension_with_xlsx_contents_uses_openpyxl(
    tmp_path: Path,
) -> None:
    """detecta conteúdo OOXML mesmo quando a extensão do arquivo é XLSB."""

    xlsx_path = tmp_path / "sample.xlsx"
    source = tmp_path / "renamed.xlsb"

    dataframe = pd.DataFrame(
        {
            "A": [1, 2],
            "B": ["x", "y"],
        }
    )

    dataframe.to_excel(xlsx_path, index=False)
    xlsx_path.rename(source)

    assert resolve_excel_engine(source, source.suffix) == "openpyxl"

    with ZipFile(source) as archive:
        assert "xl/workbook.xml" in archive.namelist()