from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.config import settings
from api.services.run_service import load_run_metrics

router = APIRouter(prefix="/runs", tags=["metrics"])


@router.get("/{run_id}/metrics")
def get_run_metrics(run_id: str) -> list[dict[str, Any]]:
    """retorna as métricas de uma execução em formato de API."""
    run_path = Path(settings.data.output_path) / settings.data.runs_folder / run_id
    return load_run_metrics(run_path)
