"""testes da fábrica de fontes de dados."""

from pathlib import Path

import pytest

from src.core.exceptions import DataLoadingError
from src.data_sources.base import BaseSource
from src.data_sources.csv_source import CsvSource
from src.data_sources.excel_source import ExcelSource
from src.data_sources.factory import DataSourceFactory
from src.data_sources.parquet_source import ParquetSource


def test_create_csv_source() -> None:
    """cria a fonte correta para arquivos CSV."""

    source = DataSourceFactory.create(
        Path("data.csv"),
    )

    assert isinstance(source, CsvSource)


@pytest.mark.parametrize(
    "extension",
    [
        ".xlsx",
        ".xlsb",
        ".xls",
    ],
)
def test_create_excel_source(extension: str) -> None:
    """cria a fonte correta para arquivos Excel."""

    source = DataSourceFactory.create(
        Path(f"data{extension}"),
    )

    assert isinstance(source, ExcelSource)


def test_create_parquet_source() -> None:
    """cria a fonte correta para arquivos Parquet."""

    source = DataSourceFactory.create(
        Path("data.parquet"),
    )

    assert isinstance(source, ParquetSource)


@pytest.mark.parametrize(
    "filename",
    [
        "data.CSV",
        "data.Csv",
        "data.XLSX",
        "data.XlSb",
        "data.PARQUET",
    ],
)
def test_create_is_case_insensitive(filename: str) -> None:
    """trata extensões de arquivo de forma insensível a maiúsculas."""

    source = DataSourceFactory.create(
        Path(filename),
    )

    assert isinstance(source, BaseSource)


def test_unsupported_extension_raises_error() -> None:
    """levanta erro para extensões não suportadas."""

    with pytest.raises(
        DataLoadingError,
        match="extensão de arquivo não suportada",
    ):
        DataSourceFactory.create(
            Path("data.json"),
        )


def test_register_custom_source() -> None:
    """permite registrar uma fonte customizada."""

    class CustomSource(BaseSource):
        """Custom source used for testing."""

        def read(
            self,
            source: Path,
        ):
            raise NotImplementedError

    extension = ".custom_test"

    DataSourceFactory.register(
        extension,
        CustomSource,
    )

    try:
        source = DataSourceFactory.create(
            Path("data.custom_test"),
        )

        assert isinstance(source, CustomSource)

    finally:
        DataSourceFactory._registry.unregister(
            extension,
        )


def test_register_normalizes_extension() -> None:
    """normaliza extensões registradas para minúsculas."""

    class CustomSource(BaseSource):
        """Custom source used for testing."""

        def read(
            self,
            source: Path,
        ):
            raise NotImplementedError

    extension = ".CUSTOM_TEST"

    DataSourceFactory.register(
        extension,
        CustomSource,
    )

    try:
        source = DataSourceFactory.create(
            Path("data.custom_test"),
        )

        assert isinstance(source, CustomSource)

    finally:
        DataSourceFactory._registry.unregister(
            extension,
        )