"""
visualizações de métricas do modelo.

este módulo fornece visualizações para auxiliar na análise
das métricas utilizadas na avaliação do modelo.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.core.exceptions.data import DataError


class MetricsVisualizer:
    """
    cria visualizações de métricas de avaliação do modelo.

    responsabilidades
    ------------------
    - validar as métricas antes da geração das visualizações.
    - visualizar métricas de avaliação em um gráfico de barras.
    - gerar visualizações de métricas a partir de um dataframe.
    """

    def plot(
        self,
        metrics: dict[str, float],
    ) -> Figure:
        """
        gera um gráfico de barras com as métricas de avaliação.

        parameters
        ----------
        metrics
            dicionário contendo os nomes e os valores das métricas.

        returns
        -------
        Figure
            figura contendo o gráfico das métricas de avaliação.
        """

        self._validate_metrics(metrics)

        names = list(metrics.keys())
        values = list(metrics.values())

        figure, axis = plt.subplots()

        axis.bar(
            names,
            values,
        )

        axis.set_title(
            "Model Metrics"
        )
        axis.set_xlabel(
            "Metric"
        )
        axis.set_ylabel(
            "Value"
        )

        figure.tight_layout()

        return figure

    def plot_dataframe(
        self,
        dataframe: pd.DataFrame,
        metric_column: str = "value",
        name_column: str = "metric",
    ) -> Figure:
        """
        gera um gráfico de métricas a partir de um dataframe.

        parameters
        ----------
        dataframe
            dataframe que será utilizado para a visualização.

        metric_column
            nome da coluna que contém os valores das métricas.

        name_column
            nome da coluna que contém os nomes das métricas.

        returns
        -------
        Figure
            figura contendo o gráfico das métricas de avaliação.

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
                "Cannot visualize empty metrics."
            )

        required = {
            metric_column,
            name_column,
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

        metrics = dict(
            zip(
                dataframe[name_column],
                dataframe[metric_column],
            )
        )

        return self.plot(metrics)

    @staticmethod
    def _validate_metrics(
        metrics: dict[str, float],
    ) -> None:
        """
        valida o dicionário de métricas utilizado para geração
        das visualizações.

        parameters
        ----------
        metrics
            dicionário contendo os nomes e os valores das métricas.

        raises
        ------
        typeerror
            se o objeto informado não for um dicionário.

        dataerror
            se o dicionário estiver vazio, possuir nomes de métricas
            inválidos ou valores não numéricos.
        """

        if not isinstance(
            metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dictionary."
            )

        if not metrics:
            raise DataError(
                "Cannot visualize empty metrics."
            )

        if any(
            not isinstance(name, str)
            or not name.strip()
            for name in metrics
        ):
            raise DataError(
                "Metric names must be non-empty strings."
            )

        if any(
            not isinstance(value, (int, float))
            for value in metrics.values()
        ):
            raise DataError(
                "Metric values must be numeric."
            )
