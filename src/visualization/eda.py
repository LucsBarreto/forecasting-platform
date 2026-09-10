"""
visualizações de análise exploratória de dados.

este módulo fornece visualizações para auxiliar na análise
exploratória do dataframe.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from src.core.exceptions.data import DataError


class EDAVisualizer:
    """
    cria visualizações para análise exploratória de dados.

    responsabilidades
    ------------------
    - validar o dataframe antes da geração das visualizações.
    - visualizar a quantidade de valores ausentes por coluna.
    - visualizar a distribuição de colunas numéricas.
    """

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        valida o dataframe utilizado para geração das visualizações.

        parameters
        ----------
        dataframe
            dataframe que será validado.

        raises
        ------
        typeerror
            se o objeto informado não for um dataframe do pandas.

        dataerror
            se o dataframe estiver vazio.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise DataError(
                "Cannot visualize an empty dataframe."
            )

    def plot_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> Figure:
        """
        gera um gráfico com a quantidade de valores ausentes por coluna.

        parameters
        ----------
        dataframe
            dataframe que será utilizado para a visualização.

        returns
        -------
        Figure
            figura contendo o gráfico de valores ausentes.
        """

        self._validate_dataframe(dataframe)

        missing = dataframe.isna().sum()

        figure, axis = plt.subplots()

        missing.plot(
            kind="bar",
            ax=axis,
        )

        axis.set_title(
            "Missing Values by Column"
        )
        axis.set_xlabel(
            "Column"
        )
        axis.set_ylabel(
            "Missing Values"
        )

        figure.tight_layout()

        return figure

    def plot_numeric_distribution(
        self,
        dataframe: pd.DataFrame,
        column: str,
        bins: int = 20,
    ) -> Figure:
        """
        gera um gráfico com a distribuição de uma coluna numérica.

        parameters
        ----------
        dataframe
            dataframe que será utilizado para a visualização.

        column
            nome da coluna numérica que será visualizada.

        bins
            quantidade de intervalos utilizada no histograma.

        returns
        -------
        Figure
            figura contendo o gráfico de distribuição.

        raises
        ------
        dataerror
            se a coluna não existir ou não for numérica.

        valueerror
            se a quantidade de intervalos for menor ou igual a zero.
        """

        self._validate_dataframe(dataframe)

        if column not in dataframe.columns:
            raise DataError(
                f"Column '{column}' was not found."
            )

        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        ):
            raise DataError(
                f"Column '{column}' must be numeric."
            )

        if bins <= 0:
            raise ValueError(
                "bins must be greater than zero."
            )

        figure, axis = plt.subplots()

        dataframe[column].plot(
            kind="hist",
            bins=bins,
            ax=axis,
        )

        axis.set_title(
            f"Distribution of {column}"
        )
        axis.set_xlabel(
            column
        )
        axis.set_ylabel(
            "Frequency"
        )

        figure.tight_layout()

        return figure
