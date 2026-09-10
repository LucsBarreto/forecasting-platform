"""
módulo de carregamento de dados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.exceptions import DataLoadingError
from src.core.logger.logger import get_logger
from src.data_sources.factory import DataSourceFactory


from .discovery import FileDiscovery


logger = get_logger()


class DataLoader:
    """carrega e consolida datasets."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self) -> pd.DataFrame:
        """
        carrega todos os datasets suportados.

        returns:
            dataframe consolidado.
        """

        files = FileDiscovery(
            self._directory
        ).discover()

        dataframes: list[pd.DataFrame] = []

        for file in files:

            logger.info(
                f"carregando '{file.name}'."
            )

            datasource = DataSourceFactory.create(
                file
            )

            dataframe = datasource.read(file)

            dataframes.append(dataframe)

        if not dataframes:
            raise DataLoadingError(
                "nenhum dataframe foi carregado"
            )

        logger.info(
            "concatenando datasets"
        )

        dataframe = pd.concat(
            dataframes,
            ignore_index=True,
            copy=False,
        )

        logger.success(
            f"dataset final: {len(dataframe):,} rows."
        )

        return dataframe