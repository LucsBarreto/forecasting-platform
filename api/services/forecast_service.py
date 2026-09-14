import json
from pathlib import Path
from typing import Any

from src.config import settings


def load_forecast_candidates(run_path: str | Path) -> list[dict[str, Any]]:
    """retorna uma lista de previsões candidatas sob uma execução."""
    path = Path(run_path)
    candidates = []

    for forecast_file in sorted(path.glob("**/*.parquet")):
        candidates.append({
            "name": forecast_file.name,
            "path": str(forecast_file.relative_to(path.parent.parent)),
            "type": "parquet",
        })

    for forecast_file in sorted(path.glob("**/*.csv")):
        candidates.append({
            "name": forecast_file.name,
            "path": str(forecast_file.relative_to(path.parent.parent)),
            "type": "csv",
        })

    return candidates
