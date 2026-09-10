"""
processador de valores ausentes.

este módulo é responsável pelo tratamento de valores ausentes
durante a etapa de pré-processamento.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.config.pipeline import MissingSettings


@dataclass(slots=True)
class MissingValueProcessor:
    """
    trata valores ausentes de acordo com as estratégias configuradas.

    responsabilidades
    ------------------
    - tratar valores ausentes em colunas numéricas.
    - tratar valores ausentes em colunas categóricas.
    - tratar valores ausentes em colunas de data e hora.
    - aplicar as estratégias de preenchimento ou remoção configuradas.
    """

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa o pipeline de tratamento de valores ausentes.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe com os valores ausentes tratados.
        """

        data = dataframe.copy()

        config: MissingSettings = settings.pipeline.preprocessing.missing

        data = self._process_numeric(data, config)
        data = self._process_categorical(data, config)
        data = self._process_datetime(data, config)

        return data

    @staticmethod
    def _process_numeric(
        dataframe: pd.DataFrame,
        config: MissingSettings,
    ) -> pd.DataFrame:
        """
        trata valores ausentes em colunas numéricas.

        operations
        ----------
        - preenche valores ausentes com a média.
        - preenche valores ausentes com a mediana.
        - preenche valores ausentes com zero.
        - preenche valores ausentes com uma constante configurada.
        - remove linhas com valores ausentes.
        """

        dataframe = dataframe.copy()

        columns = dataframe.select_dtypes(include="number").columns

        for column in columns:
            strategy = config.numeric_strategy

            if strategy == "mean":
                dataframe[column] = dataframe[column].fillna(
                    dataframe[column].mean()
                )

            elif strategy == "median":
                dataframe[column] = dataframe[column].fillna(
                    dataframe[column].median()
                )

            elif strategy == "zero":
                dataframe[column] = dataframe[column].fillna(0)

            elif strategy == "constant":
                dataframe[column] = dataframe[column].fillna(
                    config.constant_value
                )

            elif strategy == "drop":
                dataframe = dataframe.dropna(subset=[column])

        return dataframe

    @staticmethod
    def _process_categorical(
        dataframe: pd.DataFrame,
        config: MissingSettings,
    ) -> pd.DataFrame:
        """
        trata valores ausentes em colunas categóricas.

        operations
        ----------
        - preenche valores ausentes com a moda.
        - preenche valores ausentes com uma constante configurada.
        - remove linhas com valores ausentes.
        """

        dataframe = dataframe.copy()

        columns = dataframe.select_dtypes(
            include=["object", "string", "category"],
        ).columns

        for column in columns:
            strategy = config.categorical_strategy

            if strategy == "mode":
                mode = dataframe[column].mode(dropna=True)

                if not mode.empty:
                    dataframe[column] = dataframe[column].fillna(
                        mode.iloc[0]
                    )

            elif strategy == "constant":
                dataframe[column] = dataframe[column].fillna(
                    config.constant_value
                )

            elif strategy == "drop":
                dataframe = dataframe.dropna(subset=[column])

        return dataframe

    @staticmethod
    def _process_datetime(
        dataframe: pd.DataFrame,
        config: MissingSettings,
    ) -> pd.DataFrame:
        """
        trata valores ausentes em colunas de data e hora.

        operations
        ----------
        - remove linhas com valores ausentes.
        - preenche valores ausentes com uma constante configurada.
        """

        dataframe = dataframe.copy()

        columns = dataframe.select_dtypes(
            include=[
                "datetime64[ns]",
                "datetimetz",
            ],
        ).columns

        for column in columns:
            strategy = config.datetime_strategy

            if strategy == "drop":
                dataframe = dataframe.dropna(subset=[column])

            elif strategy == "constant":
                dataframe[column] = dataframe[column].fillna(
                    config.constant_value
                )

        return dataframe
