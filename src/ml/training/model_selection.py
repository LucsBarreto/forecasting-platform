"""
seleção de modelos baseada na configuração da aplicação.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.ml.factory import ModelFactory


@dataclass(slots=True)
class BacktestModelSelector:
    """
    seleciona o modelo vencedor a partir de uma estrutura de
    BacktestResult por candidato, sem entrar na lógica de geração de folds.

    policy
    -----
    - aceita um dicionário de nome->BacktestResult;
    - usa uma métrica explícita e uma direção explícita: minimize|maximize;
    - retorna o nome do modelo vencedor.
    """

    metric: str = "metric"
    objective: str = "minimize"

    def select(self, results: dict[str, object]) -> str:
        """seleciona o modelo com melhor agregação da métrica informada."""
        if not isinstance(results, dict) or not results:
            raise ValueError("results must be a non-empty dictionary of BacktestResult objects.")
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise ValueError("metric must be a non-empty string.")
        if self.objective not in {"minimize", "maximize"}:
            raise ValueError("objective must be either 'minimize' or 'maximize'.")

        ranked = []
        for model_name, result in results.items():
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model names must be non-empty strings.")
            if not hasattr(result, "aggregated_metrics"):
                raise TypeError("each result must expose aggregated_metrics.")
            if self.metric not in result.aggregated_metrics:
                raise ValueError(f"Metric '{self.metric}' was not found in aggregated metrics.")

            value = float(result.aggregated_metrics[self.metric]["mean"])
            ranked.append((value, model_name))

        if self.objective == "minimize":
            ranked.sort(key=lambda item: item[0])
        else:
            ranked.sort(key=lambda item: item[0], reverse=True)

        return ranked[0][1]


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