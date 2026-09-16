"""Paginação agnóstica de armazenamento para dados de forecast."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import HTTPException

from src.api.schemas.forecast import ForecastDataResponse, ForecastPagination

MAX_FORECAST_LIMIT = 1000
DEFAULT_FORECAST_LIMIT = 100


def validate_forecast_pagination(offset: int, limit: int) -> None:
    """Valida os parâmetros de paginação compartilhados pelo service e pelo contrato."""
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be greater than or equal to zero")
    if limit <= 0 or limit > MAX_FORECAST_LIMIT:
        raise HTTPException(status_code=422, detail=f"limit must be between 1 and {MAX_FORECAST_LIMIT}")


def paginate_forecast_rows(
    run_id: str,
    target: str,
    rows: list[dict[str, Any]],
    offset: int = 0,
    limit: int = DEFAULT_FORECAST_LIMIT,
) -> ForecastDataResponse:
    """Ordena cronologicamente e pagina registros sem conhecer CSV, Parquet ou pandas."""
    validate_forecast_pagination(offset, limit)

    def chronological_key(row: dict[str, Any]) -> tuple[int, str]:
        value = row.get("date")
        if isinstance(value, (datetime, date)):
            return (0, value.isoformat())
        return (1, str(value))

    ordered_rows = sorted(rows, key=chronological_key)
    page = ordered_rows[offset : offset + limit]
    return ForecastDataResponse(
        run_id=run_id,
        target=target,
        data=page,
        pagination=ForecastPagination(
            offset=offset,
            limit=limit,
            returned=len(page),
            has_next=offset + len(page) < len(ordered_rows),
        ),
    )
