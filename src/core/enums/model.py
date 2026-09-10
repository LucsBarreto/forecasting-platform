"""
modelos de machine learning suportados pela plataforma.
"""

from enum import StrEnum


class ModelType(StrEnum):
    """modelos de machine learning suportados."""

    LINEAR_REGRESSION = "linear_regression"

    RANDOM_FOREST = "random_forest"

    XGBOOST = "xgboost"

    LIGHTGBM = "lightgbm"

    CATBOOST = "catboost"

    PROPHET = "prophet"

    SARIMA = "sarima"