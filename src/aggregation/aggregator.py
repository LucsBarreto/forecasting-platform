"""
módulo de agregação de dados.

agrega dados transacionais na granularidade temporal definida pela configuração do pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.config.pipeline import AggregationSettings
from src.core.exceptions import DataValidationError


@dataclass(slots=True)
class Aggregator:
    """
    agrega dados transacionais conforme a configuração do pipeline.
    """

    def aggregate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        agrega o dataframe conforme a configuração definida.

        parametros
        ----------
        dataframe
            dataframe transacional.

        retorna
        -------
        pd.dataframe
            dataframe agregado.
        """

        data = dataframe.copy()

        config: AggregationSettings = (
            settings.pipeline.preprocessing.aggregation
        )

        self._validate_columns(
            dataframe=dataframe,
            config=config,
        )

        aggregation = (
            data.groupby(
                config.group_by,
                dropna=False,
                observed=False,
            )
            .agg(config.metrics)
            .reset_index()
        )

        return aggregation

    @staticmethod
    def _validate_columns(
        dataframe: pd.DataFrame,
        config: AggregationSettings,
    ) -> None:
        """
        valida se todas as colunas configuradas existem no dataframe.
        """

        required_columns = (
            set(config.group_by)
            | set(config.metrics.keys())
        )

        missing_columns = sorted(
            required_columns - set(dataframe.columns)
        )

        if missing_columns:
            raise DataValidationError(
                (
                    "Missing required columns for aggregation: "
                    f"{', '.join(missing_columns)}"
                )
            )