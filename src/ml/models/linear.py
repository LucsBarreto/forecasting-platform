"""
modelo de regressão linear.

este módulo fornece uma implementação de regressão linear
para a plataforma de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel


@dataclass(slots=True)
class LinearRegressionModel(BaseModel):
    """
    wrapper para o modelo de regressão linear.

    responsabilidades
    ------------------
    - validar os dados de treinamento.
    - treinar o modelo de regressão linear.
    - gerar previsões.
    - expor o estado de treinamento.
    """

    model: LinearRegression | None = None

    def __post_init__(self) -> None:
        """inicializa o modelo de regressão subjacente."""

        if self.model is None:
            self.model = LinearRegression()

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "linear_regression"

    @property
    def is_fitted(self) -> bool:
        """retorna se o modelo foi treinado."""

        return hasattr(
            self.model,
            "coef_",
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> LinearRegressionModel:
        """
        treina o modelo de regressão linear.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        returns
        -------
        linearregressionmodel
            instância do modelo treinado.
        """

        self._validate_training_data(
            X,
            y,
        )

        self.model.fit(
            X,
            y,
        )

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
            features para previsão.

        returns
        -------
        pd.series
            previsões do modelo.
        """

        self._validate_prediction_data(
            X,
        )

        predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    @staticmethod
    def _validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treinamento."""

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
                "Training features and target must have "
                "the same number of rows."
            )

        if X.isna().any().any():
            raise DataValidationError(
                "Training features contain missing values."
            )

        if y.isna().any():
            raise DataValidationError(
                "Training target contains missing values."
            )

    def _validate_prediction_data(
        self,
        X: pd.DataFrame,
    ) -> None:
        """valida os dados de previsão."""

        if not self.is_fitted:
            raise DataValidationError(
                "Linear regression model has not been fitted."
            )

        if X.empty:
            raise DataValidationError(
                "Prediction features cannot be empty."
            )

        if X.isna().any().any():
            raise DataValidationError(
                "Prediction features contain missing values."
            )