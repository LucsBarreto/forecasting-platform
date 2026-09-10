"""
visualizações de previsão.

este módulo fornece visualizações para auxiliar na análise
de valores históricos e previstos ao longo do tempo.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.core.exceptions.data import DataError


class ForecastVisualizer:
    """
    cria visualizações de valores históricos e previstos.

    responsabilidades
    ------------------
    - validar o dataframe antes da geração das visualizações.
    - validar as colunas necessárias para a visualização.
    - converter e ordenar os valores da coluna de datas.
    - visualizar valores reais e previstos ao longo do tempo.
    """

    def plot(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        actual_column: str,
        prediction_column: str,
    ) -> Figure:
        """
        gera um gráfico com valores reais e previstos ao longo do tempo.

        parameters
        ----------
        dataframe
            dataframe que será utilizado para a visualização.

        date_column
            nome da coluna que contém as datas utilizadas no eixo temporal.

        actual_column
            nome da coluna que contém os valores reais.

        prediction_column
            nome da coluna que contém os valores previstos.

        returns
        -------
        Figure
            figura contendo o gráfico de valores reais e previstos.

        raises
        ------
        dataerror
            se alguma das colunas necessárias não existir ou se a coluna
            de datas não contiver valores de data válidos.
        """

        self._validate_dataframe(dataframe)

        required = {
            date_column,
            actual_column,
            prediction_column,
        }

        missing = (
            required
            - set(dataframe.columns)
        )

        if missing:
            raise DataError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

        data = dataframe.copy()

        data[date_column] = pd.to_datetime(
            data[date_column],
            errors="coerce",
            format="mixed"
        )

        if data[date_column].isna().all():
            raise DataError(
                f"Column '{date_column}' does not contain valid dates."
            )

        data = data.sort_values(
            date_column,
            kind="stable",
        )

        figure, axis = plt.subplots()

        axis.plot(
            data[date_column],
            data[actual_column],
            label="Actual",
        )

        axis.plot(
            data[date_column],
            data[prediction_column],
            label="Prediction",
        )

        axis.set_title(
            "Forecast"
        )
        axis.set_xlabel(
            "Date"
        )
        axis.set_ylabel(
            "Value"
        )
        axis.legend()

        figure.tight_layout()

        return figure

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """
        valida o dataframe utilizado para geração da visualização
        de previsão.

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
                "Cannot visualize an empty forecast dataframe."
            )
