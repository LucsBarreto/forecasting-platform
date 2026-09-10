"""
modelo de regressão random forest.

este módulo implementa o modelo random forest utilizado
para previsão de alvos numéricos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted

from src.core.exceptions.validation import DataValidationError
from src.ml.models.base_model import BaseModel

@dataclass(slots=True)
class RandomForestModel(BaseModel):
    """
    modelo de regressão random forest.

    parameters
    ----------
    n_estimators
        número de árvores na floresta.
    max_depth
        profundidade máxima de cada árvore.
    min_samples_split
        número mínimo de amostras necessário para dividir um nó.
    min_samples_leaf
        número mínimo de amostras necessário em uma folha.
    max_features
        número de features consideradas ao buscar a melhor divisão.
    random_state
        semente aleatória para reprodutibilidade.
    n_jobs
        número de tarefas paralelas.
    """

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: int | float | str | None = 1.0
    random_state: int = 42
    n_jobs: int = -1

    model: RandomForestRegressor = field(
        init=False,
        repr=False,
    )

    _is_fitted: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """inicializa o modelo random forest."""

        self._validate_configuration()

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    @property
    def name(self) -> str:
        """retorna o nome do modelo."""

        return "random_forest"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> RandomForestModel:
        """
        treina o modelo random forest.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        returns
        -------
        randomforestmodel
            instância do modelo treinado.
        """

        self._validate_training_data(
            X,
            y,
        )

        self.model.fit(
            X,
            y,
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """gera previsões."""

        self._validate_prediction_data(X)

        if not self._is_fitted:
            raise DataValidationError(
                "Random Forest model has not been fitted."
            )

        predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    def get_params(self) -> dict[str, object]:
        """retorna os parâmetros do modelo random forest."""

        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }

    def _validate_configuration(self) -> None:
        """valida a configuração de random forest."""

        if self.n_estimators <= 0:
            raise DataValidationError(
                "n_estimators must be greater than zero."
            )

        if self.min_samples_split < 2:
            raise DataValidationError(
                "min_samples_split must be at least 2."
            )

        if self.min_samples_leaf <= 0:
            raise DataValidationError(
                "min_samples_leaf must be greater than zero."
            )

        if self.max_depth is not None and self.max_depth <= 0:
            raise DataValidationError(
                "max_depth must be greater than zero."
            )

        if self.n_jobs == 0:
            raise DataValidationError(
                "n_jobs cannot be zero."
            )

    @staticmethod
    def _validate_training_data(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de treino."""

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
        """valida os dados de previsão."""

        if not isinstance(X, pd.DataFrame):
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
