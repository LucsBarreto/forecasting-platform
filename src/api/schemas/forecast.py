"""Schemas estáveis para leitura paginada de forecasts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ForecastPagination(BaseModel):
    """Metadados da página retornada ao consumidor HTTP."""

    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    returned: int = Field(ge=0)
    has_next: bool


class ForecastDataResponse(BaseModel):
    """Envelope público para uma página de dados de forecast."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    target: str
    data: list[dict[str, Any]]
    pagination: ForecastPagination
