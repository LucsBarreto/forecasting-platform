from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel
from src.ml.models.feature_adapter import adapt_features


@dataclass(slots=True)
class Forecaster:
    """
    gera previsões usando um modelo de previsão.

    o modelo pode ser fornecido na inicialização ou
    diretamente ao método predict().
    """

    model: BaseModel | None = None

    def predict(
        self,
        *args: Any,
    ) -> pd.Series:
        """
        gera previsões.

        interfaces suportadas
        ---------------------
        predict(features)
            utiliza o modelo fornecido na inicialização.

        predict(model, features)
            utiliza o modelo fornecido durante a previsão.
        """

        model, features = self._resolve_arguments(args)

        self._validate_model(model)
        self._validate_features(features)

        predictions = model.predict(
            adapt_features(model, features)
        )

        if not isinstance(
            predictions,
            pd.Series,
        ):
            predictions = pd.Series(
                predictions,
                index=features.index,
            )

        if len(predictions) != len(features):
            raise DataValidationError(
                "Number of predictions does not match "
                "the number of input features."
            )

        predictions = predictions.copy()
        predictions.index = features.index

        return predictions

    def _resolve_arguments(
        self,
        args: tuple[Any, ...],
    ) -> tuple[BaseModel, pd.DataFrame]:
        """resolve o modelo e as features a partir dos argumentos fornecidos."""

        if len(args) == 1:
            if self.model is None:
                raise DataValidationError(
                    "A model is required for forecasting."
                )

            return self.model, args[0]

        if len(args) == 2:
            model, features = args

            return model, features

        raise TypeError(
            "predict() expects either "
            "(features) or (model, features)."
        )

    @staticmethod
    def _validate_model(
        model: BaseModel | None,
    ) -> None:
        """valida o modelo utilizado para previsão."""

        if model is None:
            raise DataValidationError(
                "A model is required for forecasting."
            )

    @staticmethod
    def _validate_features(
        features: pd.DataFrame,
    ) -> None:
        """valida as features utilizadas para previsão."""

        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            raise DataValidationError(
                "Prediction features cannot be empty."
            )