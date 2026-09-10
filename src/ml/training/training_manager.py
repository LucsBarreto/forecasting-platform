"""
orquestração de treinamento.

este módulo coordena a seleção, criação e treinamento dos modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ml.factory import ModelFactory
from src.ml.models.base_model import BaseModel

from .model_selection import ModelSelector
from .trainer import ModelTrainer


@dataclass(slots=True)
class TrainingResult:
    """
    resultado de uma execução de treinamento de modelo.

    attributes
    ----------
    models
        dicionário contendo os modelos treinados indexados pelo nome.
    """

    models: dict[str, BaseModel]


@dataclass(slots=True)
class TrainingManager:
    """
    orquestra o treinamento dos modelos de machine learning configurados.

    responsabilidades
    ------------------
    - resolver os modelos configurados para treinamento.
    - criar instâncias dos modelos através do modelfactory.
    - treinar os modelos através do modeltrainer.
    - retornar os modelos treinados.

    esta classe não realiza:
    - pré-processamento de dados;
    - engenharia de features;
    - seleção de features;
    - divisão temporal;
    - avaliação de modelos.
    """

    selector: ModelSelector

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> TrainingResult:
        """
        treina todos os modelos resolvidos pelo seletor de modelos.

        parameters
        ----------
        x
            features de treinamento.

        y
            alvo de treinamento.

        model_params
            parâmetros opcionais específicos de cada modelo.

        returns
        -------
        trainingresult
            modelos treinados indexados pelo nome.
        """

        self._validate_input(
            X,
            y,
        )

        params = model_params or {}

        fitted_models: dict[str, BaseModel] = {}

        for model_name in self.selector.resolve():
            model = ModelFactory.create(
                model_name,
                **params.get(
                    model_name,
                    {},
                ),
            )

            trainer = ModelTrainer(
                model=model,
            )

            fitted_model = trainer.train(
                X,
                y,
            )

            fitted_models[
                model_name
            ] = fitted_model

        return TrainingResult(
            models=fitted_models,
        )

    @staticmethod
    def _validate_input(
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """valida os dados de entrada para treinamento."""

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