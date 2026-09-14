from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.config import settings
from api.services.forecast_service import load_forecast_candidates

router = APIRouter(prefix="/runs", tags=["forecasts"])


@router.get("/{run_id}/forecasts")
def get_forecasts(run_id: str) -> list[dict[str, Any]]:
    """retorna os arquivos de previsão encontrados para uma execução."""
    run_path = Path(settings.data.output_path) / settings.data.runs_folder / run_id
    return load_forecast_candidates(run_path)
