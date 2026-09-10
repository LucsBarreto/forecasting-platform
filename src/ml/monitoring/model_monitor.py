"""
monitoramento de modelos.

este módulo fornece informações de monitoramento das
execuções de modelos de previsão.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.core.exceptions.validation import DataValidationError


@dataclass(slots=True)
class ModelMonitor:
    """
    monitora informações de execução dos modelos.

    responsabilidades
    ------------------
    - registrar metadados de execução dos modelos.
    - registrar métricas de avaliação.
    - validar informações de monitoramento.
    - disponibilizar o histórico de monitoramento.
    """

    _history: list[dict[str, Any]] = field(
        default_factory=list,
    )

    def record(
        self,
        *,
        model_name: str,
        metrics: dict[str, float],
        n_observations: int,
    ) -> dict[str, Any]:
        """
        registra uma execução de modelo.

        parameters
        ----------
        model_name
            nome do modelo avaliado.

        metrics
            métricas de avaliação.

        n_observations
            número de observações utilizadas na avaliação.

        returns
        -------
        dict[str, any]
            informações da execução registrada.
        """

        self._validate_model_name(
            model_name,
        )

        self._validate_metrics(
            metrics,
        )

        self._validate_observations(
            n_observations,
        )

        record = {
            "timestamp": datetime.now(
                timezone.utc,
            ),
            "model_name": model_name,
            "metrics": metrics.copy(),
            "n_observations": n_observations,
        }

        self._history.append(
            record,
        )

        return record.copy()

    def history(self) -> pd.DataFrame:
        """
        retorna o histórico de monitoramento como dataframe.

        returns
        -------
        pd.dataframe
            histórico das execuções monitoradas.
        """

        if not self._history:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "model_name",
                    "metrics",
                    "n_observations",
                ],
            )

        return pd.DataFrame(
            self._history,
        )

    def latest(self) -> dict[str, Any] | None:
        """
        retorna o registro de monitoramento mais recente.

        returns
        -------
        dict[str, any] | none
            registro mais recente ou none quando o histórico estiver vazio.
        """

        if not self._history:
            return None

        return self._history[-1].copy()

    def clear(self) -> None:
        """remove todo o histórico de monitoramento."""

        self._history.clear()

    @staticmethod
    def _validate_model_name(
        model_name: str,
    ) -> None:
        """valida o nome do modelo."""

        if not isinstance(
            model_name,
            str,
        ):
            raise TypeError(
                "model_name must be a string."
            )

        if not model_name.strip():
            raise DataValidationError(
                "Model name cannot be empty."
            )

    @staticmethod
    def _validate_metrics(
        metrics: dict[str, float],
    ) -> None:
        """valida as métricas de avaliação."""

        if not isinstance(
            metrics,
            dict,
        ):
            raise TypeError(
                "metrics must be a dictionary."
            )

        if not metrics:
            raise DataValidationError(
                "Metrics cannot be empty."
            )

        for name, value in metrics.items():

            if not isinstance(
                name,
                str,
            ):
                raise TypeError(
                    "Metric names must be strings."
                )

            if not isinstance(
                value,
                (int, float),
            ):
                raise TypeError(
                    f"Metric '{name}' must be numeric."
                )

        if any(
            pd.isna(value)
            for value in metrics.values()
        ):
            raise DataValidationError(
                "Metrics cannot contain null values."
            )

    @staticmethod
    def _validate_observations(
        n_observations: int,
    ) -> None:
        """valida a quantidade de observações."""

        if not isinstance(
            n_observations,
            int,
        ):
            raise TypeError(
                "n_observations must be an integer."
            )

        if n_observations <= 0:
            raise DataValidationError(
                "Number of observations must be greater than zero."
            )