"""
engenharia de features de lag.

este módulo cria features baseadas em valores defasados para modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.pipeline import LagFeatureSettings
from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class LagFeatureEngineer:
    """
    cria features baseadas em valores defasados.

    responsabilidades
    ------------------
    - validar a configuração.
    - validar o dataframe.
    - ordenar o dataframe.
    - criar lags dos alvos.
    - criar lags agrupados.
    """

    config: LagFeatureSettings

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa a engenharia de features de lag.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe com as features de lag.
        """

        data = dataframe.copy()

        if not self.config.enabled:
            return data

        self._validate_configuration(
            self.config,
        )

        self._validate_dataframe(
            data,
            self.config,
        )

        data = self._sort_dataframe(
            data,
            self.config,
        )

        return self._create_lags(
            data=data,
            config=self.config,
        )

    @staticmethod
    def _validate_configuration(
        config: LagFeatureSettings,
    ) -> None:
        """valida a configuração das features de lag."""

        if not config.target_columns:
            raise DataValidationError(
                "No target columns were configured."
            )

        if not config.group_levels:
            raise DataValidationError(
                "No grouping levels were configured."
            )

        if not config.lags:
            raise DataValidationError(
                "No lag values were configured."
            )

        if any(
            lag <= 0
            for lag in config.lags
        ):
            raise DataValidationError(
                "Lag values must be greater than zero."
            )

        if len(config.lags) != len(set(config.lags)):
            raise DataValidationError(
                "Duplicated lag values are not allowed."
            )

        for group_level in config.group_levels:
            if not group_level:
                raise DataValidationError(
                    "Grouping levels cannot be empty."
                )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        config: LagFeatureSettings,
    ) -> None:
        """valida as colunas obrigatórias do dataframe."""

        required_columns = {
            config.date_column,
            *config.target_columns,
        }

        for group_level in config.group_levels:
            required_columns.update(group_level)

        missing_columns = sorted(
            required_columns - set(dataframe.columns),
        )

        if missing_columns:
            raise DataValidationError(
                "Missing columns: "
                + ", ".join(missing_columns),
            )

        if not pd.api.types.is_datetime64_any_dtype(
            dataframe[config.date_column],
        ):
            raise DataValidationError(
                (
                    f"Date column "
                    f"'{config.date_column}' "
                    "must be datetime."
                ),
            )

    @staticmethod
    def _sort_dataframe(
        dataframe: pd.DataFrame,
        config: LagFeatureSettings,
    ) -> pd.DataFrame:
        """ordena o dataframe pelas colunas de agrupamento e data."""

        sort_columns: list[str] = []

        for group_level in config.group_levels:
            for column in group_level:
                if column not in sort_columns:
                    sort_columns.append(column)

        if config.date_column not in sort_columns:
            sort_columns.append(
                config.date_column,
            )

        return (
            dataframe
            .sort_values(
                by=sort_columns,
                kind="stable",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _create_lags(
        data: pd.DataFrame,
        config: LagFeatureSettings,
    ) -> pd.DataFrame:
        """
        cria features de lag para todos os alvos e níveis
        de agrupamento configurados.
        """

        data = data.copy()

        for group_level in config.group_levels:

            group_suffix = "_".join(
                group_level,
            )

            grouped = data.groupby(
                group_level,
                sort=False,
                dropna=False,
            )

            for target in config.target_columns:

                for lag in config.lags:

                    feature_name = (
                        f"{target}_lag_"
                        f"{lag}_"
                        f"{group_suffix}"
                    )

                    data[feature_name] = (
                        grouped[target]
                        .shift(lag)
                    )

        return data