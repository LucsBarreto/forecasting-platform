"""
seleção de modelos baseada na configuração da aplicação.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.ml.factory import ModelFactory


@dataclass(slots=True)
class ModelSelector:
    """
    resolve os modelos que devem participar do treinamento.

    responsabilidades
    ------------------
    - ler as configurações de ativação dos modelos.
    - resolver os modelos registrados e habilitados.
    - respeitar a configuração do automl.
    """

    def available_models(self) -> tuple[str, ...]:
        """retorna todos os modelos registrados."""

        return ModelFactory.available_models()

    def enabled_models(self) -> tuple[str, ...]:
        """retorna os modelos habilitados na configuração da aplicação."""

        configured_models = settings.models.models

        enabled = [
            model_name
            for model_name in self.available_models()
            if self._is_enabled(
                configured_models,
                model_name,
            )
        ]

        return tuple(enabled)

    def resolve(self) -> tuple[str, ...]:
        """
        resolve os modelos que devem ser treinados.

        quando o automl está habilitado, retorna todos os modelos
        configurados e habilitados.

        quando o automl está desabilitado, retorna o primeiro modelo
        habilitado registrado na configuração.
        """

        enabled = self.enabled_models()

        if not enabled:
            raise ValueError(
                "No machine learning models are enabled."
            )

        if settings.models.automl.enabled:
            return enabled

        return (
            enabled[0],
        )

    @staticmethod
    def _is_enabled(
        configured_models,
        model_name: str,
    ) -> bool:
        """verifica se um modelo está habilitado na configuração."""

        model_settings = getattr(
            configured_models,
            model_name,
            None,
        )

        if model_settings is None:
            return False

        return model_settings.enabled