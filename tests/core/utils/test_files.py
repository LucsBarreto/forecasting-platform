"""
Tests for file utilities.
"""

from pathlib import Path

import pytest

from src.core.utils.files import find_files


def test_find_files_returns_matching_files(
    tmp_path: Path,
) -> None:
    """Test that matching files are returned."""

    (tmp_path / "data_01.xlsx").touch()
    (tmp_path / "data_02.xlsx").touch()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert result == [
        tmp_path / "data_01.xlsx",
        tmp_path / "data_02.xlsx",
    ]


def test_find_files_respects_pattern(
    tmp_path: Path,
) -> None:
    """Test that only files matching the pattern are returned."""

    (tmp_path / "data.xlsx").touch()
    (tmp_path / "data.csv").touch()
    (tmp_path / "data.parquet").touch()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert result == [
        tmp_path / "data.xlsx",
    ]


def test_find_files_returns_path_objects(
    tmp_path: Path,
) -> None:
    """Test that results are Path objects."""

    (tmp_path / "data.xlsx").touch()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert len(result) == 1
    assert isinstance(result[0], Path)


def test_find_files_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Test that files are returned in sorted order."""

    (tmp_path / "data_03.xlsx").touch()
    (tmp_path / "data_01.xlsx").touch()
    (tmp_path / "data_02.xlsx").touch()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert result == [
        tmp_path / "data_01.xlsx",
        tmp_path / "data_02.xlsx",
        tmp_path / "data_03.xlsx",
    ]


def test_find_files_returns_empty_list_when_no_file_matches(
    tmp_path: Path,
) -> None:
    """Test that no matches return an empty list."""

    (tmp_path / "data.csv").touch()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert result == []


def test_find_files_ignores_directories(
    tmp_path: Path,
) -> None:
    """Test that directories are not returned as files."""

    (tmp_path / "data.xlsx").touch()
    (tmp_path / "other.xlsx").mkdir()

    result = find_files(
        tmp_path,
        "*.xlsx",
    )

    assert result == [
        tmp_path / "data.xlsx",
    ]


def test_find_files_raises_when_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    """Test that a missing directory raises FileNotFoundError."""

    directory = tmp_path / "does_not_exist"

    with pytest.raises(
        FileNotFoundError,
        match="diretório não encontrado",
    ):
        find_files(
            directory,
            "*.xlsx",
        )


def test_find_files_raises_when_path_is_not_directory(
    tmp_path: Path,
) -> None:
    """Test that a file path raises NotADirectoryError."""

    file_path = tmp_path / "data.xlsx"
    file_path.touch()

    with pytest.raises(
        NotADirectoryError,
        match="não é um diretório",
    ):
        find_files(
            file_path,
            "*.xlsx",
        )