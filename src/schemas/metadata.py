"""Contrato validável dos metadados de um forecast exportado."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ForecastPeriod(BaseModel):
    """Período temporal coberto pelo forecast."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date

    @model_validator(mode="after")
    def validate_order(self) -> ForecastPeriod:
        if self.end < self.start:
            raise ValueError("period.end must be greater than or equal to period.start")
        return self


class MetadataTimestamps(BaseModel):
    """Timestamps timezone-aware relevantes à execução."""

    model_config = ConfigDict(extra="forbid")

    started_at: datetime
    finished_at: datetime

    @field_validator("started_at", "finished_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> MetadataTimestamps:
        if self.finished_at < self.started_at:
            raise ValueError("timestamps.finished_at must be after timestamps.started_at")
        return self


class MetadataSchema(BaseModel):
    """Contrato mínimo de identidade e auditoria do forecast."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^RUN_\d{8}_\d{6}$")
    target: Literal["VOLUME", "VALOR"]
    model: str = Field(min_length=1)
    horizon: int = Field(gt=0)
    period: ForecastPeriod
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    timestamps: MetadataTimestamps
    forecast_rows: int | None = Field(default=None, ge=0)
    final_metric: float | None = None

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value
