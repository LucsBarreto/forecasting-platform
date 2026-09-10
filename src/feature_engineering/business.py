"""
engenharia de features de negócio.

este módulo cria features orientadas ao negócio para modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.pipeline import BusinessFeatureSettings
from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class BusinessFeatureEngineer:
    """
    cria features de negócio para modelos de previsão.

    responsabilidades
    ------------------
    - validar a configuração.
    - validar o dataframe.
    - criar indicadores comerciais.
    - evitar vazamento do alvo.
    """

    config: BusinessFeatureSettings

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """executa a engenharia de features de negócio."""

        data = dataframe.copy()

        if not self.config.enabled:
            return data

        self._validate_configuration(
            self.config
        )

        self._validate_dataframe(
            data,
            self.config,
        )

        data = self._create_features(
            data,
            self.config,
        )

        return data

    @staticmethod
    def _validate_configuration(
        config: BusinessFeatureSettings,
    ) -> None:
        """valida a configuração das features de negócio."""

        if not config.target_columns:
            raise DataValidationError(
                "No target columns were configured."
            )

        if not config.group_levels:
            raise DataValidationError(
                "No grouping levels were configured."
            )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        config: BusinessFeatureSettings,
    ) -> None:
        """valida as colunas obrigatórias do dataframe."""

        required_columns = {
            config.date_column,
            *config.target_columns,
        }

        for level in config.group_levels:
            required_columns.update(level)

        missing = sorted(
            required_columns - set(dataframe.columns)
        )

        if missing:
            raise DataValidationError(
                "Missing columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _create_features(
        data: pd.DataFrame,
        config: BusinessFeatureSettings,
    ) -> pd.DataFrame:
        """cria as features de negócio configuradas."""

        data = data.copy()

        data = BusinessFeatureEngineer._create_growth_features(
            data,
            config,
        )

        data = BusinessFeatureEngineer._create_share_features(
            data,
            config,
        )

        return data

    @staticmethod
    def _create_growth_features(
        data: pd.DataFrame,
        config: BusinessFeatureSettings,
    ) -> pd.DataFrame:
        """
        cria features históricas de crescimento.

        o valor atual é excluído para evitar vazamento.
        """

        data = data.copy()

        for group_level in config.group_levels:

            suffix = "_".join(group_level)

            for target in config.target_columns:

                previous = (
                    data.groupby(
                        group_level,
                        sort=False,
                    )[target]
                    .shift(1)
                )

                feature_name = (
                    f"{target}_growth_{suffix}"
                )

                data[feature_name] = (
                    data[target]
                    .sub(previous)
                    .div(previous.abs())
                    .replace(
                        [float("inf"), -float("inf")],
                        pd.NA,
                    )
                )

        return data

    @staticmethod
    def _create_share_features(
        data: pd.DataFrame,
        config: BusinessFeatureSettings,
    ) -> pd.DataFrame:
        """cria features de participação do alvo dentro de cada grupo configurado."""

        data = data.copy()

        for group_level in config.group_levels:

            suffix = "_".join(group_level)

            for target in config.target_columns:

                group_total = data.groupby(
                    group_level,
                    sort=False,
                )[target].transform("sum")

                feature_name = (
                    f"{target}_share_{suffix}"
                )

                data[feature_name] = (
                    data[target]
                    .div(group_total)
                    .replace(
                        [float("inf"), -float("inf")],
                        pd.NA,
                    )
                )

        return data