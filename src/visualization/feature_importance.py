"""
visualizações de importância das features.

este módulo fornece visualizações para auxiliar na análise
da importância das features utilizadas pelo modelo.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.core.exceptions.data import DataError


class FeatureImportanceVisualizer:
    """
    cria visualizações de importância das features.

    responsabilidades
    ------------------
    - validar o dataframe antes da geração das visualizações.
    - validar as colunas necessárias para a visualização.
    - visualizar a importância das features em um gráfico de barras.
    """

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        valida o dataframe utilizado para geração da visualização
        de importância das features.

        parameters
        ----------
        dataframe
            dataframe que será validado.

        raises
        ------
        typeerror
            se o objeto informado não for um dataframe do pandas.

        dataerror
            se o dataframe estiver vazio ou não possuir as colunas
            necessárias para a visualização.
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
                "Cannot visualize empty feature importance."
            )

        required_columns = {
            "feature",
            "importance",
        }

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise DataError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

    def plot(
        self,
        dataframe: pd.DataFrame,
        top_n: int | None = None,
    ) -> Figure:
        """
        gera um gráfico de barras com a importância das features.

        parameters
        ----------
        dataframe
            dataframe que será utilizado para a visualização.

        top_n
            quantidade máxima de features que será visualizada.
            se não informado, todas as features serão utilizadas.

        returns
        -------
        Figure
            figura contendo o gráfico de importância das features.

        raises
        ------
        valueerror
            se a quantidade de features for menor ou igual a zero.
        """

        self._validate_dataframe(dataframe)

        if top_n is not None and top_n <= 0:
            raise ValueError(
                "top_n must be greater than zero."
            )

        data = dataframe.copy()

        data = data.sort_values(
            "importance",
            ascending=True,
            kind="stable",
        )

        if top_n is not None:
            data = data.tail(top_n)

        figure, axis = plt.subplots()

        axis.barh(
            data["feature"],
            data["importance"],
        )

        axis.set_title(
            "Feature Importance"
        )
        axis.set_xlabel(
            "Importance"
        )
        axis.set_ylabel(
            "Feature"
        )

        figure.tight_layout()

        return figure
