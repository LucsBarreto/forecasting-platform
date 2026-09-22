"""Contrato validável das métricas persistidas de uma execução."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetricsSchema(BaseModel):
    """Envelope oficial do artefato `metrics.json` em formato multimodelo."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^RUN_\d{8}_\d{6}$")
    target: Literal["VOLUME", "VALOR"]
    metrics: dict[str, float]

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metric_mapping(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("metrics must be a non-empty object")

        normalized: dict[str, float] = {}
        for name, metric_value in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"metric name must be a non-empty string: {name!r}")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise ValueError(f"metric '{name}' must be numeric")
            metric_number = float(metric_value)
            if not math.isfinite(metric_number):
                raise ValueError(f"metric '{name}' must be finite")
            normalized[name] = metric_number

        return normalized


class SingleModelMetricsSchema(BaseModel):
    """Envelope legado para um único modelo, mantido apenas por compatibilidade."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^RUN_\d{8}_\d{6}$")
    target: Literal["VOLUME", "VALOR"]
    model: str = Field(min_length=1)
    metrics: dict[str, float]

    @field_validator("model")
    @classmethod
    def reject_blank_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be blank")
        return value

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metric_mapping(cls, value: Any) -> dict[str, float]:
        if not isinstance(value, dict) or not value:
            raise ValueError("metrics must be a non-empty object")

        normalized: dict[str, float] = {}
        for name, metric_value in value.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"metric name must be a non-empty string: {name!r}")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise ValueError(f"metric '{name}' must be numeric")
            metric_number = float(metric_value)
            if not math.isfinite(metric_number):
                raise ValueError(f"metric '{name}' must be finite")
            normalized[name] = metric_number

        return normalized
