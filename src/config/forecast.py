"""
configurações relacionadas à previsão.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ForecastHorizonSettings(BaseModel):
    """configuração do horizonte de previsão."""

    minimum: int = Field(gt=0)

    maximum: int = Field(gt=0)


class ValidationSettings(BaseModel):
    """configuração da validação temporal."""

    train_size: float = Field(
        gt=0,
        lt=1,
    )

    validation_size: float = Field(
        gt=0,
        lt=1,
    )

    test_size: float = Field(
        gt=0,
        lt=1,
    )


class ForecastSettings(BaseModel):
    """configuração da previsão."""

    targets: list[str]

    forecast_horizon: ForecastHorizonSettings

    validation: ValidationSettings

    random_seed: int