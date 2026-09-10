"""
modelo de previsão catboost.

este módulo fornece uma implementação catboost compatível
com a interface comum BaseModel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel


@dataclass(slots=True)
class CatBoostModel(BaseModel):
    """
    modelo de regressão catboost.

    parameters
    ----------
    params
        parâmetros passados diretamente para o catboostregressor.
    """

    params: dict[str, Any] = field(
        default_factory=dict,
    )

    _model: CatBoostRegressor = field(
        init=False,
    )

    _is_fitted: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """inicializa o estimador catboost."""

        self._model = CatBoostRegressor(
            **self.params,
        )

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "catboost"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> CatBoostModel:
        """treina o modelo catboost."""

        self._validate_training_data(
            X,
            y,
        )

        self._model.fit(
            X,
            y,
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """gera previsões usando o modelo catboost."""

        if not self._is_fitted:
            raise DataValidationError(
                "CatBoost model has not been fitted."
            )

        self._validate_prediction_data(
            X,
        )

        predictions = self._model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    def get_params(self) -> dict[str, Any]:
        """retorna os parâmetros do modelo catboost."""

        return self._model.get_params()

    @staticmethod
    def _validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treino."""

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise DataValidationError(
                "Training features must be a pandas DataFrame."
            )

        if not isinstance(
            y,
            pd.Series,
        ):
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

        if X.isnull().any().any():
            raise DataValidationError(
                "Training features cannot contain null values."
            )

        if y.isnull().any():
            raise DataValidationError(
                "Training target cannot contain null values."
            )

    @staticmethod
    def _validate_prediction_data(
        X: pd.DataFrame,
    ) -> None:
        """valida as features de previsão."""

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise DataValidationError(
                "Prediction features must be a pandas DataFrame."
            )

        if X.empty:
            raise DataValidationError(
                "Prediction features cannot be empty."
            )

        if X.isnull().any().any():
            raise DataValidationError(
                "Prediction features cannot contain null values."
            )