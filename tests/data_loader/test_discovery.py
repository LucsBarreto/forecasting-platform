"""
Tests for file discovery.
"""

from pathlib import Path

import pytest

from src.core.exceptions import DataLoadingError
from src.data_loader.discovery import FileDiscovery


def test_discover_returns_supported_files(
    tmp_path: Path,
) -> None:
    """Test that supported files are discovered."""

    (tmp_path / "data.xlsx").touch()
    (tmp_path / "data.csv").touch()
    (tmp_path / "data.parquet").touch()

    files = FileDiscovery(tmp_path).discover()

    assert len(files) == 3

    assert tmp_path / "data.xlsx" in files
    assert tmp_path / "data.csv" in files
    assert tmp_path / "data.parquet" in files


def test_discover_ignores_unsupported_files(
    tmp_path: Path,
) -> None:
    """Test that unsupported extensions are ignored."""

    (tmp_path / "data.xlsx").touch()
    (tmp_path / "data.json").touch()
    (tmp_path / "data.txt").touch()

    files = FileDiscovery(tmp_path).discover()

    assert files == [
        tmp_path / "data.xlsx",
    ]


def test_discover_returns_sorted_files(
    tmp_path: Path,
) -> None:
    """Test that files are returned in sorted order."""

    (tmp_path / "c.xlsx").touch()
    (tmp_path / "a.xlsx").touch()
    (tmp_path / "b.xlsx").touch()

    files = FileDiscovery(tmp_path).discover()

    assert files == [
        tmp_path / "a.xlsx",
        tmp_path / "b.xlsx",
        tmp_path / "c.xlsx",
    ]


def test_discover_raises_when_directory_does_not_exist(
    tmp_path: Path,
) -> None:
    """Test missing directory handling."""

    directory = tmp_path / "does_not_exist"

    with pytest.raises(
        DataLoadingError,
        match="diretório não encontrado",
    ):
        FileDiscovery(directory).discover()


def test_discover_raises_when_path_is_not_directory(
    tmp_path: Path,
) -> None:
    """Test that a file cannot be used as a directory."""

    file = tmp_path / "data.xlsx"
    file.touch()

    with pytest.raises(
        DataLoadingError,
        match="não é um diretório",
    ):
        FileDiscovery(file).discover()


def test_discover_raises_when_no_supported_files_exist(
    tmp_path: Path,
) -> None:
    """Test empty supported file discovery."""

    (tmp_path / "data.txt").touch()
    (tmp_path / "data.json").touch()

    with pytest.raises(
        DataLoadingError,
        match="nenhum arquivo com extensão suportada",
    ):
        FileDiscovery(tmp_path).discover()