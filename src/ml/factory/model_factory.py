"""
factory de modelos.

este módulo centraliza a criação de modelos de previsão.
"""

from __future__ import annotations

from typing import Any

from src.ml.models.base_model import BaseModel
from src.ml.registry import ModelRegistry


class ModelFactory:
    """
    cria instâncias de modelos de previsão.

    a descoberta e o registro dos modelos são delegados ao
    ModelRegistry.
    """

    @classmethod
    def create(
        cls,
        model_name: str,
        **kwargs: Any,
    ) -> BaseModel:
        """cria uma instância de modelo."""

        model_class = ModelRegistry.get(
            model_name,
        )

        return model_class(**kwargs)

    @classmethod
    def available_models(
        cls,
    ) -> tuple[str, ...]:
        """retorna os nomes de todos os modelos registrados."""

        return ModelRegistry.available_models()

    @classmethod
    def is_registered(
        cls,
        model_name: str,
    ) -> bool:
        """verifica se um modelo está registrado."""

        return ModelRegistry.is_registered(
            model_name,
        )