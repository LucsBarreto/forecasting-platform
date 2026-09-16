"""Service de leitura de métricas do run para a camada HTTP."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from src.config import settings


class MetricsService:
    """Lê o artefato de métricas de um run já persistido sem recalcular nada."""

    def get_metrics(self, run_id: str) -> dict[str, dict[str, float]]:
        """Retorna o envelope JSON normalizado de métricas."""
        runs_root = Path(settings.data.output_path) / settings.data.runs_folder
        run_path = runs_root / run_id
        if not run_path.exists() or not run_path.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        metrics_path = run_path / "metrics" / "metrics.json"
        if not metrics_path.exists():
            raise HTTPException(status_code=404, detail="metrics not found")

        with metrics_path.open(mode="r", encoding="utf-8") as file:
            metrics = json.load(file)

        return {"metrics": metrics}
