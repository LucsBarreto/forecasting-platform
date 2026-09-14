from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.config import settings
from api.services.run_service import list_runs, load_run_metadata

router = APIRouter(prefix="/runs", tags=["runs"])


@router.get("")
def get_runs() -> list[dict[str, Any]]:
    """lista as execuções publicadas em outputs/runs."""
    return list_runs()


@router.get("/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """retorna o metadata.json de uma execução específica."""
    run_path = Path(settings.data.output_path) / settings.data.runs_folder / run_id
    return load_run_metadata(run_path)
