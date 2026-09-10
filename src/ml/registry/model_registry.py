"""
registro de modelos.

este módulo mantém a coleção de modelos de previsão disponíveis.
"""

from __future__ import annotations

from typing import Type

from src.ml.models.base_model import BaseModel
from src.ml.models.baseline import BaselineModel
from src.ml.models.catboost import CatBoostModel
from src.ml.models.lightgbm import LightGBMModel
from src.ml.models.linear import LinearRegressionModel
from src.ml.models.prophet import ProphetModel
from src.ml.models.random_forest import RandomForestModel
from src.ml.models.sarima import SarimaModel
from src.ml.models.xgboost import XGBoostModel


class ModelRegistry:
    """
    registro contendo todos os modelos de previsão disponíveis.

    o registro é responsável apenas pelo cadastro e consulta
    dos modelos. a instanciação dos modelos é realizada pelo modelfactory.
    """

    _models: dict[str, Type[BaseModel]] = {
        "baseline": BaselineModel,
        "linear_regression": LinearRegressionModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "prophet": ProphetModel,
        "sarima": SarimaModel,
    }

    @classmethod
    def get(
        cls,
        model_name: str,
    ) -> Type[BaseModel]:
        """
        retorna a classe do modelo registrada com o nome informado.

        parameters
        ----------
        model_name
            nome do modelo registrado.

        returns
        -------
        type[basemodel]
            classe do modelo registrada.

        raises
        ------
        valueerror
            se o modelo não estiver registrado.
        """

        normalized_name = cls._normalize_name(
            model_name,
        )

        model_class = cls._models.get(
            normalized_name,
        )

        if model_class is None:
            available = ", ".join(
                cls.available_models()
            )

            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available}"
            )

        return model_class

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseModel],
    ) -> None:
        """
        registra uma classe de modelo.

        parameters
        ----------
        name
            nome utilizado para identificar o modelo.

        model_class
            classe de modelo que implementa basemodel.

        raises
        ------
        typeerror
            se model_class não herdar de basemodel.

        valueerror
            se o nome estiver vazio ou já estiver registrado.
        """

        normalized_name = cls._normalize_name(
            name,
        )

        if not issubclass(
            model_class,
            BaseModel,
        ):
            raise TypeError(
                "model_class must inherit from BaseModel."
            )

        if normalized_name in cls._models:
            raise ValueError(
                f"Model '{normalized_name}' is already registered."
            )

        cls._models[normalized_name] = model_class

    @classmethod
    def available_models(
        cls,
    ) -> tuple[str, ...]:
        """retorna os nomes de todos os modelos registrados."""

        return tuple(
            cls._models.keys()
        )

    @classmethod
    def is_registered(
        cls,
        model_name: str,
    ) -> bool:
        """verifica se um modelo está registrado."""

        normalized_name = cls._normalize_name(
            model_name,
        )

        return normalized_name in cls._models

    @staticmethod
    def _normalize_name(
        model_name: str,
    ) -> str:
        """normaliza o nome de um modelo."""

        if not isinstance(
            model_name,
            str,
        ):
            raise TypeError(
                "model_name must be a string."
            )

        normalized = model_name.strip().lower()

        if not normalized:
            raise ValueError(
                "model_name cannot be empty."
            )

        return normalized