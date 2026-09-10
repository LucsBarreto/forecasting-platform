"""
modelo de previsão baseline.

este módulo implementa um modelo ingênuo de previsão usando
o último valor observado do alvo.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel


@dataclass(slots=True)
class BaselineModel(BaseModel):
    """
    modelo ingênuo de previsão.

    armazena o último valor observado do alvo e o utiliza
    como previsão para novas observações.

    em previsões agrupadas, utiliza o último valor observado
    para cada grupo configurado.
    """

    date_column: str = "DATA"
    target_column: str = "VOLUME"
    group_columns: list[str] | None = None

    _last_values: dict[tuple, float] | None = None
    _global_last_value: float | None = None
    _fitted: bool = False

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "baseline"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaselineModel:
        """
        treina o modelo baseline.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        returns
        -------
        baselinemodel
            instância do modelo treinado.
        """

        self._validate_training_data(
            X,
            y,
        )

        data = X.copy()
        data["_target"] = y.to_numpy()

        if self.date_column not in data.columns:
            raise DataValidationError(
                f"Date column '{self.date_column}' was not found."
            )

        if self.group_columns:
            self._validate_group_columns(data)

            data = data.sort_values(
                by=[
                    *self.group_columns,
                    self.date_column,
                ],
                kind="stable",
            )

            grouped = data.groupby(
                self.group_columns,
                sort=False,
            )["_target"]

            last_values = grouped.last()

            self._last_values = {
                self._normalize_group_key(key): float(value)
                for key, value in last_values.items()
            }

        else:
            data = data.sort_values(
                by=self.date_column,
                kind="stable",
            )

            self._global_last_value = float(
                data["_target"].iloc[-1]
            )

        self._fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        gera previsões baseline.

        parameters
        ----------
        x
            features para previsão.

        returns
        -------
        pd.series
            previsões do modelo baseline.
        """

        if not self._fitted:
            raise DataValidationError(
                "Baseline model must be fitted before prediction."
            )

        data = X.copy()

        if self.group_columns:
            self._validate_group_columns(data)

            if self._last_values is None:
                raise DataValidationError(
                    "Grouped baseline values were not initialized."
                )

            predictions = data.apply(
                lambda row: self._get_group_prediction(row),
                axis=1,
            )

        else:
            if self._global_last_value is None:
                raise DataValidationError(
                    "Baseline value was not initialized."
                )

            predictions = pd.Series(
                self._global_last_value,
                index=data.index,
            )

        return pd.Series(
            predictions,
            index=data.index,
            name="prediction",
            dtype=float,
        )

    def get_params(self) -> dict[str, object]:
        """retorna os parâmetros do modelo."""

        return {
            "date_column": self.date_column,
            "target_column": self.target_column,
            "group_columns": self.group_columns,
        }

    def _get_group_prediction(
        self,
        row: pd.Series,
    ) -> float:
        """retorna a previsão armazenada para um grupo."""

        if self._last_values is None:
            raise DataValidationError(
                "Grouped baseline values were not initialized."
            )

        key = self._normalize_group_key(
            tuple(
                row[column]
                for column in self.group_columns or []
            )
        )

        if key not in self._last_values:
            raise DataValidationError(
                f"No baseline value found for group: {key}"
            )

        return self._last_values[key]

    def _validate_group_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """valida as colunas de agrupamento configuradas."""

        missing = [
            column
            for column in self.group_columns or []
            if column not in dataframe.columns
        ]

        if missing:
            raise DataValidationError(
                "Missing grouping columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _normalize_group_key(
        key: tuple | object,
    ) -> tuple:
        """normaliza as chaves de agrupamento do pandas para tuplas."""

        if isinstance(key, tuple):
            return key

        return (key,)

    @staticmethod
    def _validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treino."""

        if X.empty:
            raise DataValidationError(
                "Training dataframe cannot be empty."
            )

        if y.empty:
            raise DataValidationError(
                "Training target cannot be empty."
            )

        if len(X) != len(y):
            raise DataValidationError(
                "Features and target must have the same length."
            )

        if y.isna().any():
            raise DataValidationError(
                "Training target cannot contain missing values."
            )