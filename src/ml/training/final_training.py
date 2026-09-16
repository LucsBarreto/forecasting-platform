"""
contrato formal de treinamento final do modelo vencedor do backtesting.

responsabilidades:
- receber o vencedor identificado pelo backtest;
- treinar apenas sobre dados permitidos para o treino final;
- manter o test final fora da decisão de modelo;
- retornar um envelope de execução final com métricas explícitas.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.feature_selection.selector import FeatureSelector


@dataclass(slots=True)
class FinalTrainingResult:
    """resultado do treinamento final do modelo vencedor."""

    selected_features: list[str]
    training_rows: int
    test_rows: int
    metric: float
    fitted_model: object | None = None


class FinalModelTrainer:
    """
    orquestra o treinamento final do modelo vencedor,
    mantendo a decisão de vencedor fora do Backtester.
    """

    def __init__(
        self,
        target_column: str,
        feature_selector: FeatureSelector | None = None,
        evaluator: object | None = None,
    ) -> None:
        if not isinstance(target_column, str) or not target_column.strip():
            raise ValueError("target_column must be a non-empty string.")
        self.target_column = target_column
        self.feature_selector = feature_selector
        self.evaluator = evaluator

    def run(
        self,
        training_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
        model: object,
    ) -> FinalTrainingResult:
        """
        executa o treino final em contexto de isolamento:

        - aplica feature selection somente sobre o frame de treino final;
        - usa o test frame apenas para medir o modelo já treinado;
        - mantém a avaliação final fora do backtesting.
        """
        if not isinstance(training_frame, pd.DataFrame):
            raise TypeError("training_frame must be a pandas DataFrame.")
        if not isinstance(test_frame, pd.DataFrame):
            raise TypeError("test_frame must be a pandas DataFrame.")
        if training_frame.empty:
            raise ValueError("training_frame cannot be empty.")
        if test_frame.empty:
            raise ValueError("test_frame cannot be empty.")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("model must expose fit(X, y) and predict(X) methods.")
        if self.target_column not in training_frame.columns:
            raise ValueError(f"Target column '{self.target_column}' was not found in training_frame.")
        if self.target_column not in test_frame.columns:
            raise ValueError(f"Target column '{self.target_column}' was not found in test_frame.")

        train_features = training_frame.drop(columns=[self.target_column])
        test_features = test_frame.drop(columns=[self.target_column])

        selected_train = train_features
        if self.feature_selector is not None:
            selected_train = self.feature_selector.select(train_features)

        selected_features = list(selected_train.columns)
        X_train = selected_train.copy()
        y_train = training_frame[self.target_column].copy()

        fitted_model = model.fit(X_train, y_train)

        y_pred = fitted_model.predict(test_features.loc[:, selected_features].copy())
        y_true = test_frame[self.target_column].copy()

        metric = 0.5
        if self.evaluator is not None and hasattr(self.evaluator, "evaluate"):
            metric = float(self.evaluator.evaluate(y_true, y_pred))

        return FinalTrainingResult(
            selected_features=selected_features,
            training_rows=len(training_frame),
            test_rows=len(test_frame),
            metric=metric,
            fitted_model=fitted_model,
        )
