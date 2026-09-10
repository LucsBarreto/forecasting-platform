"""
factory de fontes de dados.

seleciona o leitor adequado de acordo com a extensão do arquivo.
"""

from __future__ import annotations

from pathlib import Path

from src.core.exceptions import DataLoadingError

from .base import BaseSource
from .csv_source import CsvSource
from .excel_source import ExcelSource
from .parquet_source import ParquetSource
from .registry import DataSourceRegistry


class DataSourceFactory:
    """cria leitores de dados conforme a extensão do arquivo."""

    _registry = DataSourceRegistry()

    @classmethod
    def register(
        cls,
        extension: str,
        reader: type[BaseSource],
    ) -> None:
        """
        registra um leitor de dados.

        parameters
        ----------
        extension
            extensão de arquivo associada ao leitor.

        reader
            classe responsável pela leitura.
        """

        cls._registry.register(
            extension,
            reader,
        )

    @classmethod
    def create(
        cls,
        source: Path,
    ) -> BaseSource:
        """
        cria o leitor adequado para uma fonte de dados.

        parameters
        ----------
        source
            caminho do arquivo de origem.

        returns
        -------
        basesource
            instância do leitor correspondente.

        raises
        ------
        dataloadingerror
            se a extensão do arquivo não for suportada.
        """

        extension = source.suffix.lower()

        try:
            reader = cls._registry.get(extension)

        except KeyError as exc:
            raise DataLoadingError(
                f"extensão de arquivo não suportada: {extension}"
            ) from exc

        return reader()


# registra os leitores de dados disponíveis
DataSourceFactory.register(".xlsx", ExcelSource)
DataSourceFactory.register(".xlsb", ExcelSource)
DataSourceFactory.register(".xls", ExcelSource)
DataSourceFactory.register(".csv", CsvSource)
DataSourceFactory.register(".parquet", ParquetSource)