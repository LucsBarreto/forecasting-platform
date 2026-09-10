"""
avaliação de modelos.

este módulo é responsável por avaliar modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class ModelEvaluator:
    """
    avalia previsões de modelos usando métricas configuradas.

    responsabilidades
    ------------------
    - validar os dados de avaliação.
    - calcular métricas de previsão.
    - retornar os resultados da avaliação.
    """

    metric: str = "mape"

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> float:
        """
        avalia as previsões usando a métrica configurada.

        parameters
        ----------
        y_true
            valores reais do alvo.

        y_pred
            valores previstos.

        returns
        -------
        float
            valor da métrica de avaliação.
        """

        self._validate_data(
            y_true,
            y_pred,
        )

        metric = self.metric.lower()

        if metric == "mape":
            return self._calculate_mape(
                y_true,
                y_pred,
            )

        if metric == "mae":
            return self._calculate_mae(
                y_true,
                y_pred,
            )

        if metric == "rmse":
            return self._calculate_rmse(
                y_true,
                y_pred,
            )

        if metric == "mse":
            return self._calculate_mse(
                y_true,
                y_pred,
            )

        raise DataValidationError(
            f"Unsupported evaluation metric: {self.metric}"
        )

    @staticmethod
    def _validate_data(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> None:
        """valida os valores reais e previstos."""

        if not isinstance(
            y_true,
            pd.Series,
        ):
            raise TypeError(
                "y_true must be a pandas Series."
            )

        if not isinstance(
            y_pred,
            pd.Series,
        ):
            raise TypeError(
                "y_pred must be a pandas Series."
            )

        if y_true.empty:
            raise DataValidationError(
                "Actual target cannot be empty."
            )

        if y_pred.empty:
            raise DataValidationError(
                "Predictions cannot be empty."
            )

        if len(y_true) != len(y_pred):
            raise DataValidationError(
                "Actual target and predictions "
                "must have the same number of values."
            )

        if not np.isfinite(
            y_true.to_numpy(dtype=float)
        ).all():
            raise DataValidationError(
                "Actual target contains invalid numeric values."
            )

        if not np.isfinite(
            y_pred.to_numpy(dtype=float)
        ).all():
            raise DataValidationError(
                "Predictions contain invalid numeric values."
            )

    @staticmethod
    def _calculate_mape(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> float:
        """
        calcula o erro percentual absoluto médio.

        valores reais iguais a zero são excluídos do cálculo.
        """

        actual = y_true.to_numpy(
            dtype=float,
        )

        predicted = y_pred.to_numpy(
            dtype=float,
        )

        mask = actual != 0

        if not mask.any():
            raise DataValidationError(
                "MAPE cannot be calculated when "
                "all actual values are zero."
            )

        return float(
            np.mean(
                np.abs(
                    (
                        actual[mask]
                        - predicted[mask]
                    )
                    / actual[mask]
                )
            )
            * 100
        )

    @staticmethod
    def _calculate_mae(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> float:
        """calcula o erro médio absoluto."""

        actual = y_true.to_numpy(
            dtype=float,
        )

        predicted = y_pred.to_numpy(
            dtype=float,
        )

        return float(
            np.mean(
                np.abs(
                    actual - predicted
                )
            )
        )

    @staticmethod
    def _calculate_mse(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> float:
        """calcula o erro quadrático médio."""

        actual = y_true.to_numpy(
            dtype=float,
        )

        predicted = y_pred.to_numpy(
            dtype=float,
        )

        return float(
            np.mean(
                np.square(
                    actual - predicted
                )
            )
        )

    @staticmethod
    def _calculate_rmse(
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> float:
        """calcula a raiz do erro quadrático médio."""

        return float(
            np.sqrt(
                ModelEvaluator._calculate_mse(
                    y_true,
                    y_pred,
                )
            )
        )