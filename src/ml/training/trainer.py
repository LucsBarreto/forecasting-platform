"""
treinamento de modelos.

este módulo é responsável pelo treinamento de modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ml.models.base_model import BaseModel
from src.ml.models.feature_adapter import adapt_features


@dataclass(slots=True)
class ModelTrainer:
    """
    treina modelos de machine learning.

    responsabilidades
    ------------------
    - validar os dados de treinamento.
    - treinar um modelo.
    - retornar o modelo treinado.
    """

    model: BaseModel

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseModel:
        """
        treina o modelo configurado.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        returns
        -------
        basemodel
            modelo treinado.
        """

        self._validate_training_data(
            X,
            y,
        )

        fitted_model = self.model.fit(
            adapt_features(self.model, X),
            y,
        )

        return fitted_model

    @staticmethod
    def _validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treinamento."""

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if not isinstance(
            y,
            pd.Series,
        ):
            raise TypeError(
                "y must be a pandas Series."
            )

        if X.empty:
            raise ValueError(
                "Training features cannot be empty."
            )

        if y.empty:
            raise ValueError(
                "Training target cannot be empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "Training features and target "
                "must have the same number of rows."
            )

        if X.isnull().all(axis=None):
            raise ValueError(
                "Training features cannot contain only null values."
            )

        if y.isnull().all():
            raise ValueError(
                "Training target cannot contain only null values."
            )