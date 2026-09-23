"""Service de descoberta de runs para a camada HTTP."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import HTTPException

from src.config import settings


class RunsService:
    """Lista runs persistidos em disco sem expor o filesystem ao router."""

    def list_runs(self) -> list[dict[str, str]]:
        """Retorna a representação HTTP/JSON dos runs conhecidos."""
        runs_root = Path(settings.data.output_path) / settings.data.runs_folder
        if not runs_root.exists():
            return []

        run_dirs = sorted(
            [path for path in runs_root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        )

        return [
            {
                "run_id": run_dir.name,
            }
            for run_dir in run_dirs
        ]

    def get_run(self, run_id: str) -> dict[str, str]:
        """Retorna o DTO de um run específico ou levanta 404 se ele não existir."""
        runs_root = (Path(settings.data.output_path) / settings.data.runs_folder).resolve()
        if not runs_root.exists():
            raise HTTPException(status_code=404, detail="run not found")

        run_path = (runs_root / run_id).resolve()
        try:
            run_path.relative_to(runs_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="run not found")
        if not run_path.exists() or not run_path.is_dir():
            raise HTTPException(status_code=404, detail="run not found")

        metadata_path = run_path / "metadata.json"
        if not metadata_path.is_file():
            raise HTTPException(status_code=422, detail="run metadata is invalid")

        try:
            with metadata_path.open(mode="r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (json.JSONDecodeError, OSError):
            raise HTTPException(status_code=422, detail="run metadata is invalid")

        required_fields = ("run_id", "status", "started_at", "finished_at")
        if (
            not isinstance(metadata, dict)
            or any(not isinstance(metadata.get(field), str) for field in required_fields)
            or metadata["run_id"] != run_id
        ):
            raise HTTPException(status_code=422, detail="run metadata is invalid")

        return {
            "run_id": metadata["run_id"],
            "status": metadata["status"],
            "started_at": metadata["started_at"],
            "finished_at": metadata["finished_at"],
        }
