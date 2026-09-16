"""Router HTTP mínimo para descoberta de forecasts persistidos."""

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.forecast import ForecastDataResponse
from src.api.services.forecast_service import ForecastService

router = APIRouter()


@router.get("/runs/{run_id}/forecasts")
def list_forecasts(run_id: str) -> dict[str, list[dict[str, str]]]:
    """Expõe os forecasts persistidos de um run como JSON representativo."""
    service = ForecastService()
    try:
        return service.list_forecasts(run_id)
    except HTTPException:
        raise


@router.get("/runs/{run_id}/forecasts/{target}")
def get_forecast(run_id: str, target: str) -> dict[str, str]:
    """Expõe um forecast persistido selecionado por target de negócio."""
    service = ForecastService()
    try:
        return service.get_forecast(run_id, target)
    except HTTPException:
        raise


@router.get(
    "/runs/{run_id}/forecasts/{target}/data",
    response_model=ForecastDataResponse,
)
def get_forecast_data(
    run_id: str,
    target: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
) -> ForecastDataResponse:
    """Entrega uma página de dados do forecast por target de negócio."""
    service = ForecastService()
    try:
        return service.get_forecast_data(
            run_id=run_id,
            target=target,
            offset=offset,
            limit=limit,
        )
    except HTTPException:
        raise
