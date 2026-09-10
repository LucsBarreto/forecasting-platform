"""
modelo de previsão sarima.

este módulo implementa um modelo sarima compatível com a
interface comum de modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel


@dataclass(slots=True)
class SarimaModel(BaseModel):
    """
    modelo de previsão sarima.

    parameters
    ----------
    order
        ordem sarima não sazonal (p, d, q).
    seasonal_order
        ordem sarima sazonal (p, d, q, s).
    trend
        especificação de tendência passada ao statsmodels.
    """

    order: tuple[int, int, int] = (1, 1, 1)

    seasonal_order: tuple[int, int, int, int] = (
        1,
        1,
        1,
        12,
    )

    trend: str | None = None

    _model: Any = field(
        default=None,
        init=False,
        repr=False,
    )

    _is_fitted: bool = field(
        default=False,
        init=False,
    )

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "sarima"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SarimaModel:
        """
        treina o modelo sarima.

        o sarima é um modelo univariado de séries temporais,
        portanto x é aceito por compatibilidade com a interface,
        mas não é utilizado.
        """

        self._validate_training_data(
            X,
            y,
        )

        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self._model = model.fit(
            disp=False,
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        gera previsões.

        a quantidade de previsões é determinada pelo número de
        linhas de x.
        """

        if not self._is_fitted:
            raise DataValidationError(
                "SARIMA model must be fitted before prediction."
            )

        if not isinstance(X, pd.DataFrame):
            raise DataValidationError(
                "Prediction features must be a pandas DataFrame."
            )

        if X.empty:
            raise DataValidationError(
                "Prediction features cannot be empty."
            )

        steps = len(X)

        predictions = self._model.forecast(
            steps=steps,
        )

        return pd.Series(
            predictions.to_numpy(),
            index=X.index,
            name="prediction",
        )

    def get_params(self) -> dict[str, Any]:
        """retorna os parâmetros do modelo sarima."""

        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
        }

    @staticmethod
    def _validate_training_data(
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
                "Training features and target must have "
                "the same number of rows."
            )

        if y.isna().any():
            raise DataValidationError(
                "Training target cannot contain missing values."
            )

        if not pd.api.types.is_numeric_dtype(y):
            raise DataValidationError(
                "Training target must be numeric."
            )

    @staticmethod
    def _validate_order(
        order: tuple[int, int, int],
        name: str = "order",
    ) -> None:
        """valida uma ordem sarima não sazonal."""

        if len(order) != 3:
            raise DataValidationError(
                f"{name} must contain exactly three values."
            )

        if any(
            not isinstance(value, int) or value < 0
            for value in order
        ):
            raise DataValidationError(
                f"{name} values must be non-negative integers."
            )

    @staticmethod
    def _validate_seasonal_order(
        seasonal_order: tuple[int, int, int, int],
    ) -> None:
        """valida uma ordem sarima sazonal."""

        if len(seasonal_order) != 4:
            raise DataValidationError(
                "seasonal_order must contain exactly four values."
            )

        if any(
            not isinstance(value, int) or value < 0
            for value in seasonal_order
        ):
            raise DataValidationError(
                "seasonal_order values must be "
                "non-negative integers."
            )

        if seasonal_order[3] == 1:
            raise DataValidationError(
                "Seasonal period must be greater than 1."
            )