"""
configurações relacionadas à engenharia de features.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TemporalFeatureSettings(BaseModel):
    """configuração de features temporais."""

    enabled: bool = True


class LagFeatureSettings(BaseModel):
    """configuração de features de lag."""

    enabled: bool = True

    periods: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 6, 12]
    )


class RollingFeatureSettings(BaseModel):
    """configuração de features rolling."""

    enabled: bool = True

    windows: list[int] = Field(
        default_factory=lambda: [3, 6, 9, 12]
    )


class TrendFeatureSettings(BaseModel):
    """configuração de features de tendência."""

    enabled: bool = True


class BusinessFeatureSettings(BaseModel):
    """configuração de features de negócio."""

    enabled: bool = True


class HolidayFeatureSettings(BaseModel):
    """configuração de features de feriados."""

    enabled: bool = True


class FeatureSelectionSettings(BaseModel):
    """configuração da seleção de features."""

    enabled: bool = True

    features: list[str] = Field(
        default_factory=list
    )


class FeatureSettings(BaseModel):
    """configuração da engenharia de features."""

    temporal: TemporalFeatureSettings

    lag: LagFeatureSettings

    rolling: RollingFeatureSettings

    trend: TrendFeatureSettings

    business: BusinessFeatureSettings

    holidays: HolidayFeatureSettings

    selection: FeatureSelectionSettings