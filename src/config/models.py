"""
configurações relacionadas aos modelos de machine learning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelEnabledSettings(BaseModel):
    """configuração de ativação do modelo."""

    enabled: bool = True


class AutoMLSettings(BaseModel):
    """configuração do AutoML."""

    enabled: bool = False


class ModelsRegistrySettings(BaseModel):
    """configuração dos modelos de machine learning disponíveis."""

    linear_regression: ModelEnabledSettings
    random_forest: ModelEnabledSettings
    xgboost: ModelEnabledSettings
    lightgbm: ModelEnabledSettings
    catboost: ModelEnabledSettings
    prophet: ModelEnabledSettings
    sarima: ModelEnabledSettings
    baseline: ModelEnabledSettings


class EvaluationSettings(BaseModel):
    """configuração da avaliação dos modelos."""

    metric: str


class ModelSettings(BaseModel):
    """configuração dos modelos de machine learning."""

    automl: AutoMLSettings

    models: ModelsRegistrySettings

    evaluation: EvaluationSettings