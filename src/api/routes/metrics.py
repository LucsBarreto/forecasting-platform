"""Router HTTP mínimo para métricas do run."""

from fastapi import APIRouter, HTTPException

from src.api.services.metrics_service import MetricsService

router = APIRouter()


@router.get("/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> dict[str, dict[str, float]]:
    """Expõe as métricas persistidas de um run já materializado pelo pipeline."""
    service = MetricsService()
    try:
        return service.get_metrics(run_id)
    except HTTPException:
        raise
