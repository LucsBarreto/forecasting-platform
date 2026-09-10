"""
excessões relacionadas ao modelo
"""

from .base import ApplicationError


class ModelingError(ApplicationError):
    """
    excessões de modelo básicas
    """


class TrainingError(ModelingError):
    default_message = "treinamento do modelo falhou"


class PredictionError(ModelingError):
    default_message = "predição falhou"