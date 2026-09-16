"""Router HTTP mínimo para runs."""

from fastapi import APIRouter, HTTPException

from src.api.services.runs_service import RunsService

router = APIRouter()


@router.get("/runs")
def list_runs() -> dict[str, list[dict[str, str]]]:
    """Expõe a coleção de runs de forma serializável em JSON."""
    service = RunsService()
    return {"runs": service.list_runs()}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, str]:
    """Expõe a representação detail JSON de um run específico, com 404 quando ausente."""
    service = RunsService()
    try:
        return service.get_run(run_id)
    except HTTPException:
        raise
