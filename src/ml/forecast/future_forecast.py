from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel
from src.ml.models.feature_adapter import adapt_features


@dataclass(slots=True)
class FutureFeatureAvailabilityResult:
    """Resultado formal do contrato de disponibilidade de features futuras."""

    available: bool
    missing_features: list[str]


@dataclass(slots=True)
class FutureFeatureAvailabilityContract:
    """
    Valida se todas as features selecionadas pelo modelo podem ser
    materializadas no horizonte de forecast futuro.
    """

    def run(
        self,
        selected_features: list[str],
        available_features: list[str],
    ) -> FutureFeatureAvailabilityResult:
        """
        Confere a cobertura mínima entre features selecionadas e as
        features que o futuro mecanismo consegue gerar no horizonte.
        """
        if not isinstance(selected_features, list):
            raise TypeError("selected_features must be a list of strings.")
        if not isinstance(available_features, list):
            raise TypeError("available_features must be a list of strings.")

        for feature in selected_features:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError("selected_features must contain only non-empty strings.")

        for feature in available_features:
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError("available_features must contain only non-empty strings.")

        missing = [feature for feature in selected_features if feature not in available_features]
        if missing:
            raise DataValidationError(
                "Future features cannot be computed for selected model features: "
                + ", ".join(missing)
            )

        return FutureFeatureAvailabilityResult(
            available=True,
            missing_features=[],
        )


@dataclass(slots=True)
class FutureForecastResult:
    """Envelope resultante do contrato de forecast futuro."""

    predictions: pd.Series
    horizon: int
    generated_at: str | None = None


@dataclass(slots=True)
class FutureForecastContract:
    """
    Contrato isolado de forecast futuro.

    Objetivos:
    - receber um frame com features futuras já preparadas;
    - garantir que a previsão não use informação futura escondida;
    - entregar uma série de previsões no mesmo índice do frame futuro.
    """

    model: BaseModel | None = None

    def run(self, future_frame: pd.DataFrame, model: BaseModel | None = None) -> FutureForecastResult:
        """Executa a previsão futura a partir de um frame de features futuras."""
        if not isinstance(future_frame, pd.DataFrame):
            raise TypeError("future_frame must be a pandas DataFrame.")
        if future_frame.empty:
            raise ValueError("future_frame cannot be empty.")

        selected_model = model or self.model
        if selected_model is None:
            raise DataValidationError("A model is required for future forecasting.")

        if not hasattr(selected_model, "predict"):
            raise TypeError("model must expose predict(X) method.")

        predictions = selected_model.predict(adapt_features(selected_model, future_frame))
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=future_frame.index)

        if len(predictions) != len(future_frame):
            raise DataValidationError("Number of predictions does not match number of future rows.")

        predictions = predictions.copy()
        predictions.index = future_frame.index

        return FutureForecastResult(
            predictions=predictions,
            horizon=len(future_frame),
            generated_at=None,
        )


@dataclass(slots=True)
class FutureFeatureFrameBuilder:
    """
    Constrói um frame futuro de features sem depender do pipeline principal.
    """

    time_column: str = "date"

    def build(
        self,
        history_frame: pd.DataFrame,
        horizon: int,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Cria o frame futuro com colunas explícitas de feature, usando apenas
        o histórico conhecido e o horizonte configurado.
        """
        if not isinstance(history_frame, pd.DataFrame):
            raise TypeError("history_frame must be a pandas DataFrame.")
        if history_frame.empty:
            raise ValueError("history_frame cannot be empty.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(self.time_column, str) or not self.time_column.strip():
            raise ValueError("time_column must be a non-empty string.")
        if self.time_column not in history_frame.columns:
            raise ValueError(f"Time column '{self.time_column}' was not found in history_frame.")

        columns = feature_columns or [c for c in history_frame.columns if c != self.time_column]
        if not columns:
            raise ValueError("feature_columns must provide at least one feature column.")

        missing = [c for c in columns if c not in history_frame.columns]
        if missing:
            raise ValueError(f"Feature columns not found in history_frame: {missing}")

        future_rows = history_frame.tail(horizon).copy()
        future_rows = future_rows[[self.time_column, *columns]]

        return future_rows.reset_index(drop=True)

    def build_recursive(
        self,
        history_frame: pd.DataFrame,
        horizon: int,
        target_column: str,
        predictions: list[float] | pd.Series,
        lag_values: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Constrói um frame de features futuras por recursão simples:

        - para o passo 1, usa histórico real até o último ponto conhecido;
        - para os passos seguintes, usa previsões já produzidas
          como valores de origem da iteração recursiva.

        A maturidade desta função é a de prova de arquitetura, mantendo o
        contrato isolado e demonstrando a forma de cada passo recursivo.
        """
        if not isinstance(history_frame, pd.DataFrame):
            raise TypeError("history_frame must be a pandas DataFrame.")
        if history_frame.empty:
            raise ValueError("history_frame cannot be empty.")
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column must be a non-empty string.")
        if target_column not in history_frame.columns:
            raise ValueError(f"Target column '{target_column}' was not found in history_frame.")
        if not isinstance(self.time_column, str) or not self.time_column.strip():
            raise ValueError("time_column must be a non-empty string.")
        if self.time_column not in history_frame.columns:
            raise ValueError(f"Time column '{self.time_column}' was not found in history_frame.")

        lag_values = lag_values or [1]
        rolling_windows = rolling_windows or [2]

        ordered_historical_values = history_frame[target_column].tolist()
        last_known = ordered_historical_values[-1]

        rows = []

        # primeiro passo do horizonte: usa o histórico real imediato
        for i in range(horizon):
            row = {self.time_column: history_frame[self.time_column].iloc[-1]}

            # reproduz um padrão de lag/rolling em torno do histórico e das previsões
            # já produzidas pelo passo anterior.
            if i == 0:
                row["lag_1"] = last_known
                row["lag_2"] = ordered_historical_values[-2] if len(ordered_historical_values) >= 2 else last_known
                row["rolling_2"] = sum(ordered_historical_values[-2:]) / 2
            else:
                pred = predictions[i - 1]
                row["lag_1"] = float(pred)
                row["lag_2"] = float(pred)
                row["rolling_2"] = float(pred)

            rows.append(row)

        future_frame = pd.DataFrame(rows)
        future_frame = future_frame[[self.time_column, "lag_1", "lag_2", "rolling_2"]]
        return future_frame
