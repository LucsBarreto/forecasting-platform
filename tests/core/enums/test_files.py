"""
Tests for FileType enum.
"""

from src.core.enums.files import FileType


def test_file_type_values() -> None:
    """Test supported file type values."""

    assert FileType.XLSX == ".xlsx"
    assert FileType.XLSB == ".xlsb"
    assert FileType.CSV == ".csv"
    assert FileType.PARQUET == ".parquet"


def test_file_type_is_str_enum() -> None:
    """Test that FileType members behave as strings."""

    assert isinstance(FileType.XLSX, str)
    assert isinstance(FileType.CSV, str)


def test_file_type_contains_expected_members() -> None:
    """Test all expected members exist."""

    assert set(FileType) == {
        FileType.XLSX,
        FileType.XLSB,
        FileType.CSV,
        FileType.PARQUET,
    }