"""
processador de features temporais.

este módulo cria features temporais a partir de colunas de data e hora.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.config.pipeline import DatetimeSettings


@dataclass(slots=True)
class DatetimeProcessor:
    """
    cria features temporais a partir de colunas de data e hora.

    responsabilidades
    ------------------
    - identificar colunas de data e hora.
    - criar features de ano, mês, trimestre e semestre.
    - criar features de semana, dia e dia da semana.
    - identificar finais de semana.
    - identificar início e fim de mês.
    """

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa o pipeline de criação de features temporais.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe enriquecido com features temporais.
        """

        data = dataframe.copy()

        config: DatetimeSettings = settings.pipeline.preprocessing.datetime

        datetime_columns = data.select_dtypes(
            include=["datetime64[ns]", "datetimetz"],
        ).columns

        for column in datetime_columns:

            data = self._create_features(
                data=data,
                column=column,
                config=config,
            )

        return data

    @staticmethod
    def _create_features(
        data: pd.DataFrame,
        column: str,
        config: DatetimeSettings,
    ) -> pd.DataFrame:
        """
        cria features temporais para uma coluna de data e hora.

        operations
        ----------
        - cria a feature de ano.
        - cria a feature de mês.
        - cria a feature de trimestre.
        - cria a feature de semestre.
        - cria a feature de semana.
        - cria a feature de dia.
        - cria a feature de dia da semana.
        - identifica se a data corresponde a um final de semana.
        - identifica se a data corresponde ao início do mês.
        - identifica se a data corresponde ao fim do mês.
        """

        data = data.copy()

        prefix = column

        if config.create_year:
            data[f"{prefix}_year"] = data[column].dt.year

        if config.create_month:
            data[f"{prefix}_month"] = data[column].dt.month

        if config.create_quarter:
            data[f"{prefix}_quarter"] = data[column].dt.quarter

        if config.create_semester:
            data[f"{prefix}_semester"] = (
                ((data[column].dt.month - 1) // 6) + 1
            )

        if config.create_week:
            data[f"{prefix}_week"] = (
                data[column].dt.isocalendar().week.astype("Int64")
            )

        if config.create_day:
            data[f"{prefix}_day"] = data[column].dt.day

        if config.create_day_of_week:
            data[f"{prefix}_day_of_week"] = data[column].dt.dayofweek

        if config.create_is_weekend:
            data[f"{prefix}_is_weekend"] = (
                data[column].dt.dayofweek >= 5
            )

        if config.create_is_month_start:
            data[f"{prefix}_is_month_start"] = (
                data[column].dt.is_month_start
            )

        if config.create_is_month_end:
            data[f"{prefix}_is_month_end"] = (
                data[column].dt.is_month_end
            )

        return data
