"""Contrato validável das métricas persistidas de uma execução."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.enums.metrics import MetricType


class MetricsSchema(BaseModel):
    """Envelope oficial do artefato `metrics.json`."""

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

        supported_metrics = {metric.value for metric in MetricType}
        normalized: dict[str, float] = {}
        for name, metric_value in value.items():
            if not isinstance(name, str) or name not in supported_metrics:
                raise ValueError(f"unsupported metric name: {name!r}")
            if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
                raise ValueError(f"metric '{name}' must be numeric")
            if not math.isfinite(float(metric_value)):
                raise ValueError(f"metric '{name}' must be finite")
            normalized[name] = float(metric_value)

        return normalized
