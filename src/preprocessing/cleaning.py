"""
processador de limpeza de dados.

este módulo contém a primeira etapa de pré-processamento,
responsável pela limpeza estrutural do dataframe de entrada.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.config.pipeline import CleaningSettings


@dataclass(slots=True)
class CleaningProcessor:
    """
    realiza a limpeza estrutural de um dataframe.

    responsabilidades
    ------------------
    - remover linhas completamente vazias.
    - remover linhas duplicadas.
    - normalizar nomes de colunas.
    - normalizar valores de texto.
    """

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa o pipeline de limpeza.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe limpo.
        """

        data = dataframe.copy()

        config: CleaningSettings = settings.pipeline.preprocessing.cleaning

        if config.remove_empty_rows:
            data = self._remove_empty_rows(data)

        if config.remove_duplicate_rows:
            data = self._remove_duplicate_rows(data)

        if config.normalize_column_names:
            data = self._clean_column_names(data)

        if config.normalize_string_values:
            data = self._clean_string_values(data)

        return data

    @staticmethod
    def _remove_empty_rows(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """remove linhas em que todos os valores estão ausentes."""

        return dataframe.dropna(how="all")

    @staticmethod
    def _remove_duplicate_rows(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """remove linhas duplicadas."""

        return dataframe.drop_duplicates()

    @staticmethod
    def _clean_column_names(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        normaliza os nomes das colunas.

        operations
        ----------
        - converte os nomes das colunas para texto.
        - remove espaços no início e no fim.
        - substitui múltiplos espaços por um único espaço.
        """

        dataframe.columns = (
            dataframe.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        return dataframe

    @staticmethod
    def _clean_string_values(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        normaliza valores de texto.

        operations
        ----------
        - remove espaços no início e no fim.
        - substitui múltiplos espaços por um único espaço.
        """

        string_columns = dataframe.select_dtypes(
            include=["object", "string"],
        ).columns

        for column in string_columns:
            dataframe[column] = (
                dataframe[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

        return dataframe