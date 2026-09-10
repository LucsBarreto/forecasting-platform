"""
pipeline de avaliação.

este módulo orquestra a avaliação dos modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ml.evaluation import ModelEvaluator

from .base_pipeline import BasePipeline


@dataclass(slots=True)
class EvaluationPipelineResult:
    """
    resultado produzido pelo pipeline de avaliação.

    attributes
    ----------
    metrics
        métricas de avaliação indexadas pelo nome do modelo.
    """

    metrics: dict[str, float]


class EvaluatePipeline(BasePipeline):
    """pipeline responsável por avaliar as previsões dos modelos."""

    def __init__(
        self,
        evaluator: ModelEvaluator,
    ) -> None:
        super().__init__()

        self.evaluator = evaluator

    @property
    def name(self) -> str:
        """retorna o nome do pipeline."""

        return "pipeline de avaliação"

    def run(
        self,
        y_true: pd.Series,
        predictions: dict[str, Any],
    ) -> EvaluationPipelineResult:
        """avalia as previsões dos modelos."""

        start = self._log_start()

        try:
            self._validate_input(
                y_true,
                predictions,
            )

            metrics: dict[str, float] = {}

            for model_name, y_pred in predictions.items():
                metrics[model_name] = float(
                    self.evaluator.evaluate(
                        y_true,
                        y_pred,
                    )
                )

            result = EvaluationPipelineResult(
                metrics=metrics,
            )

            self._log_finish(start)

            return result

        except Exception as exc:
            self._log_failure(exc)
            raise

    @staticmethod
    def _validate_input(
        y_true: pd.Series,
        predictions: dict[str, Any],
    ) -> None:
        """valida as entradas de avaliação."""

        if not isinstance(
            y_true,
            pd.Series,
        ):
            raise TypeError(
                "y_true must be a pandas Series."
            )

        if y_true.empty:
            raise ValueError(
                "y_true cannot be empty."
            )

        if not isinstance(
            predictions,
            dict,
        ):
            raise TypeError(
                "predictions must be a dictionary."
            )

        if len(predictions) == 0:
            raise ValueError(
                "predictions cannot be empty."
            )