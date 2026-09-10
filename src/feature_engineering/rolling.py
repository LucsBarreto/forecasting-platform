"""
engenharia de features de rolling.

este módulo cria features baseadas em janelas móveis para modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.pipeline import RollingFeatureSettings
from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class RollingFeatureEngineer:
    """
    cria features baseadas em janelas móveis.

    responsabilidades
    ------------------
    - validar a configuração.
    - validar o dataframe.
    - ordenar o dataframe cronologicamente.
    - criar features agrupadas de rolling.
    - evitar vazamento do alvo.
    """

    config: RollingFeatureSettings

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """executa a engenharia de features de rolling."""

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

        data = self._create_rolling_features(
            data,
            config,
        )

        return data

    @staticmethod
    def _validate_configuration(
        config: RollingFeatureSettings,
    ) -> None:
        """valida a configuração das features de rolling."""

        if not config.target_columns:
            raise DataValidationError(
                "No target columns were configured."
            )

        if not config.group_levels:
            raise DataValidationError(
                "No grouping levels were configured."
            )

        if not config.windows:
            raise DataValidationError(
                "No rolling windows were configured."
            )

        if any(
            window <= 0
            for window in config.windows
        ):
            raise DataValidationError(
                "Rolling windows must be greater than zero."
            )

        if len(config.windows) != len(set(config.windows)):
            raise DataValidationError(
                "Duplicated rolling windows are not allowed."
            )

        if not config.functions:
            raise DataValidationError(
                "No rolling functions were configured."
            )

        allowed_functions = {
            "mean",
            "std",
            "min",
            "max",
            "sum",
        }

        invalid_functions = (
            set(config.functions) - allowed_functions
        )

        if invalid_functions:
            raise DataValidationError(
                "Unsupported rolling functions: "
                + ", ".join(sorted(invalid_functions))
            )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
        config: RollingFeatureSettings,
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
        config: RollingFeatureSettings,
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
    def _create_rolling_features(
        data: pd.DataFrame,
        config: RollingFeatureSettings,
    ) -> pd.DataFrame:
        """
        cria features agrupadas de rolling.

        o valor atual do alvo é excluído da janela
        para evitar vazamento do alvo.
        """

        data = data.copy()

        for group_level in config.group_levels:

            group_suffix = "_".join(group_level)

            for target in config.target_columns:

                shifted = (
                    data.groupby(
                        group_level,
                        sort=False,
                    )[target]
                    .shift(1)
                )

                temp = data[
                    group_level
                ].copy()

                temp["_value"] = shifted

                grouped = temp.groupby(
                    group_level,
                    sort=False,
                )["_value"]

                for window in config.windows:

                    rolling = grouped.rolling(
                        window=window,
                        min_periods=1,
                    )

                    for function in config.functions:

                        feature_name = (
                            f"{target}_rolling_"
                            f"{function}_{window}_"
                            f"{group_suffix}"
                        )

                        values = getattr(
                            rolling,
                            function,
                        )

                        result = (
                            values()
                            .reset_index(
                                level=list(
                                    range(len(group_level))
                                ),
                                drop=True,
                            )
                        )

                        data[feature_name] = (
                            result.to_numpy()
                        )

        return data