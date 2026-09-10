from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class FeatureSelector:
    """
    seleciona features configuradas para machine learning.

    quando nenhuma feature é configurada explicitamente, todas
    as colunas do dataframe de entrada são selecionadas.
    """

    features: list[str]

    def select(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        seleciona as features configuradas.

        quando nenhuma feature é configurada explicitamente, todas
        as colunas do dataframe são selecionadas.
        """

        self._validate_dataframe(dataframe)

        if not self.features:
            return dataframe.copy()

        self._validate_features()

        missing = [
            feature
            for feature in self.features
            if feature not in dataframe.columns
        ]

        if missing:
            raise DataValidationError(
                "Missing features: "
                + ", ".join(missing)
            )

        return dataframe.loc[:, self.features].copy()

    def _validate_features(self) -> None:
        """valida os nomes das features configuradas."""

        if any(
            not isinstance(feature, str)
            or not feature.strip()
            for feature in self.features
        ):
            raise DataValidationError(
                "Feature names must be non-empty strings."
            )

        if len(self.features) != len(set(self.features)):
            raise DataValidationError(
                "Duplicated features are not allowed."
            )

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """valida o dataframe de entrada."""

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError(
                "Cannot select features from an empty dataframe."
            )