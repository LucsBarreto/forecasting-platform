"""
engenharia de features de tendência.

este módulo cria features baseadas em tendência para modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.pipeline import TrendFeatureSettings
from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class TrendFeatureEngineer:
    """
    cria features baseadas em tendência.

    responsabilidades
    ------------------
    - validar a configuração.
    - validar o dataframe.
    - ordenar o dataframe.
    - criar diferenças defasadas.
    - criar variações percentuais.
    """

    config: TrendFeatureSettings

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """executa a engenharia de features de tendência."""

        data = dataframe.copy()

        config = self.config

        if not config.enabled:
            return data

        self._validate_configuration(config)
        self._validate_dataframe(data, config)

        data = self._sort_dataframe(
            data,
            config,
        )

        data = self._create_trend_features(
            data,
            config,
        )

        return data

    @staticmethod
    def _validate_configuration(
        config: TrendFeatureSettings,
    ) -> None:
        """valida a configuração das features de tendência."""

        if not config.target_columns:
            raise DataValidationError(
                "No target columns were configured."
            )

        if not config.group_levels:
            raise DataValidationError(
                "No grouping levels were configured."
            )

        if not config.periods:
            raise DataValidationError(
                "No trend periods were configured."
            )

        if any(
            period <= 0
            for period in config.periods
        ):
            raise DataValidationError(
                "Trend periods must be greater than zero."
            )

        if len(config.periods) != len(set(config.periods)):
            raise DataValidationError(
                "Duplicated trend periods are not allowed."
            )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        config: TrendFeatureSettings,
    ) -> None:
        """valida as colunas obrigatórias do dataframe."""

        required_columns = set(
            config.target_columns
        )

        required_columns.add(
            config.date_column
        )

        for level in config.group_levels:
            required_columns.update(level)

        missing = sorted(
            column
            for column in required_columns
            if column not in dataframe.columns
        )

        if missing:
            raise DataValidationError(
                "Missing columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _sort_dataframe(
        dataframe: pd.DataFrame,
        config: TrendFeatureSettings,
    ) -> pd.DataFrame:
        """ordena o dataframe pelas colunas de agrupamento e data."""

        sort_columns: list[str] = []

        for level in config.group_levels:
            for column in level:
                if column not in sort_columns:
                    sort_columns.append(column)

        if config.date_column not in sort_columns:
            sort_columns.append(
                config.date_column
            )

        return dataframe.sort_values(
            by=sort_columns,
            kind="stable",
        ).reset_index(drop=True)

    @staticmethod
    def _create_trend_features(
        data: pd.DataFrame,
        config: TrendFeatureSettings,
    ) -> pd.DataFrame:
        """
        cria features de tendência.

        a observação atual nunca é usada como valor de referência,
        evitando vazamento do alvo.
        """

        data = data.copy()

        for group_level in config.group_levels:

            group_suffix = "_".join(group_level)

            grouped = data.groupby(
                group_level,
                sort=False,
            )

            for target in config.target_columns:

                current = data[target]

                for period in config.periods:

                    previous = grouped[target].shift(
                        period
                    )

                    difference_name = (
                        f"{target}_trend_diff_"
                        f"{period}_{group_suffix}"
                    )

                    growth_name = (
                        f"{target}_trend_growth_"
                        f"{period}_{group_suffix}"
                    )

                    data[difference_name] = (
                        current - previous
                    )

                    data[growth_name] = (
                        (current - previous)
                        / previous.abs()
                    ).where(
                        previous != 0
                    )

        return data