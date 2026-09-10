"""
modelo de previsão prophet.

este módulo implementa um modelo de previsão baseado no prophet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel


class ProphetModel(BaseModel):
    """
    modelo de previsão baseado no prophet.

    o prophet espera:
    - uma coluna de data como variável temporal.
    - uma variável alvo numérica.

    colunas adicionais não são utilizadas atualmente como regressoras.
    """

    def __init__(
        self,
        date_column: str = "DATA",
        **params: Any,
    ) -> None:
        self.date_column = date_column
        self.params = params

        self.model = None
        self._is_fitted = False

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "prophet"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ProphetModel:
        """
        treina o modelo prophet.

        parameters
        ----------
        x
            features de treinamento. deve conter a coluna de data configurada.

        y
            alvo de treinamento.

        returns
        -------
        prophetmodel
            instância do modelo treinado.
        """

        self._validate_training_data(X, y)

        try:
            from prophet import Prophet
        except ImportError as exc:
            raise DataValidationError(
                "Prophet is not installed."
            ) from exc

        training_data = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    X[self.date_column]
                ).reset_index(drop=True),
                "y": pd.to_numeric(
                    y,
                    errors="coerce",
                ).reset_index(drop=True),
            }
        )

        if training_data["y"].isna().any():
            raise DataValidationError(
                "Training target contains invalid numeric values."
            )

        model_params = dict(self.params)

        self.model = Prophet(
            **model_params,
        )

        self.model.fit(training_data)

        self._is_fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        gera previsões.

        parameters
        ----------
        x
            dataframe de previsão contendo a coluna de data configurada.

        returns
        -------
        pd.series
            valores previstos preservando o índice original do dataframe.
        """

        if not self._is_fitted or self.model is None:
            raise DataValidationError(
                "Prophet model must be fitted before prediction."
            )

        self._validate_prediction_data(X)

        future = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    X[self.date_column]
                ).reset_index(drop=True),
            }
        )

        forecast = self.model.predict(future)

        predictions = pd.Series(
            forecast["yhat"].to_numpy(),
            index=X.index,
            name="prediction",
        )

        return predictions

    def get_params(self) -> dict[str, Any]:
        """retorna os parâmetros do modelo prophet."""

        return {
            "date_column": self.date_column,
            **self.params,
        }

    def _validate_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treinamento."""

        if not isinstance(X, pd.DataFrame):
            raise DataValidationError(
                "Training features must be a pandas DataFrame."
            )

        if not isinstance(y, pd.Series):
            raise DataValidationError(
                "Training target must be a pandas Series."
            )

        if X.empty:
            raise DataValidationError(
                "Training features cannot be empty."
            )

        if y.empty:
            raise DataValidationError(
                "Training target cannot be empty."
            )

        if len(X) != len(y):
            raise DataValidationError(
                "Training features and target must have the same length."
            )

        if self.date_column not in X.columns:
            raise DataValidationError(
                f"Date column '{self.date_column}' was not found."
            )

        if X[self.date_column].isna().any():
            raise DataValidationError(
                "Training date column contains missing values."
            )

        if not pd.api.types.is_datetime64_any_dtype(
            X[self.date_column]
        ):
            try:
                pd.to_datetime(
                    X[self.date_column],
                    errors="raise",
                )
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"Date column '{self.date_column}' must contain valid dates."
                ) from exc

    def _validate_prediction_data(
        self,
        X: pd.DataFrame,
    ) -> None:
        """valida os dados de previsão."""

        if not isinstance(X, pd.DataFrame):
            raise DataValidationError(
                "Prediction features must be a pandas DataFrame."
            )

        if X.empty:
            raise DataValidationError(
                "Prediction features cannot be empty."
            )

        if self.date_column not in X.columns:
            raise DataValidationError(
                f"Date column '{self.date_column}' was not found."
            )

        if X[self.date_column].isna().any():
            raise DataValidationError(
                "Prediction date column contains missing values."
            )

        try:
            pd.to_datetime(
                X[self.date_column],
                errors="raise",
            )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Date column '{self.date_column}' must contain valid dates."
            ) from exc