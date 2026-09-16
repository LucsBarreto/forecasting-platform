"""Service de descoberta de artefatos de forecast persistidos em um run."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from src.api.schemas.forecast import ForecastDataResponse, ForecastPagination
from src.api.services.forecast_artifact_reader import ForecastArtifactReader
from src.api.services.forecast_pagination import (
    DEFAULT_FORECAST_LIMIT,
    validate_forecast_pagination,
)
from src.config import settings


class ForecastService:
    """Descobre arquivos de forecast persistidos pelo exporter e devolve DTOs de JSON."""

    _TARGET_TOKENS = {
        "VOLUME": "volume",
        "VALOR": "valor",
    }

    def __init__(self, artifact_reader: ForecastArtifactReader | None = None) -> None:
        self._artifact_reader = artifact_reader or ForecastArtifactReader()

    def list_forecasts(self, run_id: str) -> dict[str, list[dict[str, str]]]:
        """Retorna a coleção de artefatos de forecast já gravados em disco."""
        run_path = self._resolve_run_path(run_id)

        forecasts_dir = run_path / "forecasts"
        if not forecasts_dir.exists():
            raise HTTPException(status_code=404, detail="forecasts not found")

        artifacts = []
        for file_path in sorted(forecasts_dir.iterdir(), key=lambda item: item.name):
            if not file_path.is_file():
                continue
            name = file_path.name.lower()
            if name.endswith(".csv") or name.endswith(".parquet"):
                target = "VOLUME" if "volume" in name else "VALOR" if "valor" in name else "UNKNOWN"
                artifacts.append(
                    {
                        "filename": file_path.name,
                        "target": target,
                    }
                )

        return {"forecasts": artifacts}

    def get_forecast(self, run_id: str, target: str) -> dict[str, str]:
        """Resolve um artefato por identidade de negócio, sem aceitar filename arbitrário."""
        artifact_path = self._resolve_forecast_path(run_id, target)
        return {
            "filename": artifact_path.name,
            "target": target.strip().upper(),
        }

    def get_forecast_data(
        self,
        run_id: str,
        target: str,
        offset: int = 0,
        limit: int = DEFAULT_FORECAST_LIMIT,
    ) -> ForecastDataResponse:
        """Orquestra resolução, leitura limitada e representação da página de forecast."""
        validate_forecast_pagination(offset, limit)
        normalized_target = target.strip().upper()
        artifact_path = self._resolve_forecast_path(run_id, normalized_target)
        try:
            page = self._artifact_reader.read_page(artifact_path, offset, limit)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="forecast artifact not found")
        except (OSError, ValueError):
            raise HTTPException(status_code=422, detail="forecast artifact could not be read")

        return ForecastDataResponse(
            run_id=run_id,
            target=normalized_target,
            data=page.rows,
            pagination=ForecastPagination(
                offset=offset,
                limit=limit,
                returned=len(page.rows),
                has_next=page.has_next,
            ),
        )

    def _resolve_forecast_path(self, run_id: str, target: str) -> Path:
        """Resolve o artefato físico depois de validar a identidade de negócio."""
        normalized_target = target.strip().upper()
        token = self._TARGET_TOKENS.get(normalized_target)
        if token is None:
            raise HTTPException(status_code=404, detail="forecast target not found")

        forecasts = self.list_forecasts(run_id)["forecasts"]
        for artifact in forecasts:
            if artifact["target"] == normalized_target and token in artifact["filename"].lower():
                return self._resolve_run_path(run_id) / "forecasts" / artifact["filename"]

        raise HTTPException(status_code=404, detail="forecast target not found")

    @staticmethod
    def _resolve_run_path(run_id: str) -> Path:
        """Resolve um run apenas dentro da raiz configurada, evitando traversal."""
        runs_root = (Path(settings.data.output_path) / settings.data.runs_folder).resolve()
        run_path = (runs_root / run_id).resolve()
        try:
            run_path.relative_to(runs_root)
        except ValueError:
            raise HTTPException(status_code=404, detail="run not found")
        if not run_path.exists() or not run_path.is_dir():
            raise HTTPException(status_code=404, detail="run not found")
        return run_path
