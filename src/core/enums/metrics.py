"""
métricas suportadas pela plataforma.
"""

from enum import StrEnum


class MetricType(StrEnum):
    """métricas de avaliação suportadas."""

    MAE = "mae"

    RMSE = "rmse"

    MAPE = "mape"

    SMAPE = "smape"

    R2 = "r2"