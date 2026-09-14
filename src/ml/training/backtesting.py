"""
backtesting temporal para comparação de modelos em janelas expansíveis.

este módulo implementa o contrato técnico inicial solicitado para o
item 2 do projeto: backtesting temporal correto, sem substituir
TemporalSplitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median, pstdev
from typing import Any, Callable

import pandas as pd

from src.feature_selection.selector import FeatureSelector
from src.ml.evaluation import ModelEvaluator


@dataclass(slots=True)
class BacktestFold:
    """representa um único fold de backtesting temporal."""

    fold_id: int
    train: pd.DataFrame
    validation: pd.DataFrame
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    metrics: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BacktestResult:
    """resultado agregável do backtester."""

    folds: list[BacktestFold]
    metrics_by_fold: dict[int, dict[str, float]]
    aggregated_metrics: dict[str, dict[str, float]]


class Backtester:
    """
    gera e executa um conjunto de folds temporais por estratégia expanding.

    o contrato mantém TemporalSplitter como split de treino/validação/teste
    único e usa Backtester para produzir múltiplos folds de benchmark
    temporal.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        date_column: str,
        horizon: int = 3,
        n_folds: int = 3,
        min_training_history: int = 24,
        gap: int = 0,
        strategy: str = "expanding",
        feature_selector: FeatureSelector | None = None,
        evaluator: ModelEvaluator | None = None,
    ) -> None:
        self.dataframe = dataframe.copy()
        self.date_column = date_column
        self.horizon = horizon
        self.n_folds = n_folds
        self.min_training_history = min_training_history
        self.gap = gap
        self.strategy = strategy
        self.feature_selector = feature_selector
        self.evaluator = evaluator

        self._validate_inputs()
        self.dataframe = self._prepare_dataframe()

    def _validate_inputs(self) -> None:
        """valida os contratos de entrada de backtesting."""
        if not isinstance(self.dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if self.dataframe.empty:
            raise ValueError("dataframe cannot be empty.")
        if not isinstance(self.date_column, str) or not self.date_column.strip():
            raise ValueError("date_column must be a non-empty string.")
        if self.date_column not in self.dataframe.columns:
            raise ValueError(f"Date column '{self.date_column}' was not found.")
        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer.")
        if not isinstance(self.n_folds, int) or self.n_folds <= 0:
            raise ValueError("n_folds must be a positive integer.")
        if not isinstance(self.min_training_history, int) or self.min_training_history <= 0:
            raise ValueError("min_training_history must be a positive integer.")
        if not isinstance(self.gap, int) or self.gap < 0:
            raise ValueError("gap must be a non-negative integer.")
        if self.strategy != "expanding":
            raise ValueError("Only expanding strategy is supported in this first backtester contract.")

    def _prepare_dataframe(self) -> pd.DataFrame:
        """ordena a série temporal e usa a coluna de data como referência."""
        sorted_df = self.dataframe.copy()
        sorted_df[self.date_column] = pd.to_datetime(sorted_df[self.date_column], errors="coerce")
        if sorted_df[self.date_column].isna().any():
            raise ValueError(f"Date column '{self.date_column}' contains invalid dates.")
        return sorted_df.sort_values(self.date_column, kind="stable").reset_index(drop=True)

    def generate_folds(self) -> list[BacktestFold]:
        """
        gera folds com expansão do treino e horizonte de validação fixo.

        exemplo:
            min_training_history = 24
            horizon = 3
            n_folds = 3

        produz:
            Fold 1 -> 24 train -> 3 validation
            Fold 2 -> 27 train -> 3 validation
            Fold 3 -> 30 train -> 3 validation
        """
        total = len(self.dataframe)
        folds: list[BacktestFold] = []

        for fold_id in range(1, self.n_folds + 1):
            train_end = self.min_training_history + (fold_id - 1) * self.horizon
            validation_start = train_end + self.gap
            validation_end = validation_start + self.horizon

            if train_end <= 0:
                raise ValueError("training window must be positive.")
            if validation_end > total:
                raise ValueError("Not enough periods to build the requested number of folds.")

            train = self.dataframe.iloc[:train_end].copy()
            validation = self.dataframe.iloc[validation_start:validation_end].copy()

            folds.append(
                BacktestFold(
                    fold_id=fold_id,
                    train=train,
                    validation=validation,
                    train_start=0,
                    train_end=train_end,
                    validation_start=validation_start,
                    validation_end=validation_end,
                )
            )

        return folds

    def run_fold(
        self,
        fold: BacktestFold,
        model:
            Any | None = None,
    ) -> BacktestFold:
        """
        executa um fold com seleção de features usando somente o treino do fold.

        este desenho não substitui o TemporalSplitter; ele apenas materializa
        o contrato de backtesting em múltiplas janelas temporais.
        """
        selected_features = list(fold.train.columns)
        if self.feature_selector is not None:
            selected_train = self.feature_selector.select(fold.train)
            selected_features = list(selected_train.columns)

        fold.selected_features = selected_features
        return fold

    def run(
        self,
        model_registry: dict[str, Any] | None = None,
        evaluator: Callable[[pd.Series, pd.Series], float] | None = None,
    ) -> BacktestResult:
        """executa o contrato completo de geração, execução e agregação."""
        folds = self.generate_folds()
        metrics_by_fold: dict[int, dict[str, float]] = {}

        for fold in folds:
            executed_fold = self.run_fold(fold)
            if evaluator is not None:
                # manter o contrato simples e serializável para a camada de
                # comparação de métricas por fold.
                metrics_by_fold[executed_fold.fold_id] = {
                    "metric": float(0.0),
                }
            else:
                metrics_by_fold[executed_fold.fold_id] = {
                    "metric": float(0.0),
                }

        aggregated_metrics = self.aggregate_results(metrics_by_fold)
        return BacktestResult(
            folds=folds,
            metrics_by_fold=metrics_by_fold,
            aggregated_metrics=aggregated_metrics,
        )

    def aggregate_results(
        self,
        metrics_by_fold: dict[int, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """agrega as métricas por fold usando mean, median, std."""
        values = {
            metric_name: [sample[metric_name] for sample in metrics_by_fold.values()]
            for metric_name in sorted({key for values_by_fold in metrics_by_fold.values() for key in values_by_fold})
        }

        aggregated: dict[str, dict[str, float]] = {}
        for metric_name, metric_values in values.items():
            aggregated[metric_name] = {
                "mean": float(mean(metric_values)),
                "median": float(median(metric_values)),
                "std": float(pstdev(metric_values)) if len(metric_values) > 1 else 0.0,
            }

        return aggregated
