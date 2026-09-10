"""
gerencia a estrutura de diretórios das execuções do pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.config import settings


class OutputManager:
    """
    gerencia a estrutura de saída de uma execução do pipeline.

    cada execução recebe seu próprio diretório:

    outputs/
    └── runs/
        └── run_yyyymmdd_hhmmss/
            ├── forecasts/
            ├── metrics/
            ├── models/
            ├── reports/
            ├── explainability/
            ├── logs/
            └── metadata.json
    """

    def __init__(self) -> None:
        self._base_path = (
            Path(settings.data.output_path)
            / settings.data.runs_folder
        )

        self._run_id = (
            f"RUN_{datetime.now():%Y%m%d_%H%M%S}"
        )

        self._run_path = self._base_path / self._run_id

    @property
    def run_id(self) -> str:
        """retorna o identificador da execução atual."""
        return self._run_id

    @property
    def run_path(self) -> Path:
        """retorna o diretório raiz da execução atual."""
        return self._run_path

    def create_run(self) -> Path:
        """
        cria a estrutura de diretórios da execução atual.

        returns
        -------
        path
            caminho do diretório criado para a execução.
        """

        directories = [
            self.get_forecasts_dir(),
            self.get_metrics_dir(),
            self.get_models_dir(),
            self.get_reports_dir(),
            self.get_explainability_dir(),
            self.get_logs_dir(),
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._create_metadata()

        return self._run_path

    def get_forecasts_dir(self) -> Path:
        """retorna o diretório de previsões."""
        return self._run_path / "forecasts"

    def get_metrics_dir(self) -> Path:
        """retorna o diretório de métricas."""
        return self._run_path / "metrics"

    def get_models_dir(self) -> Path:
        """retorna o diretório de modelos."""
        return self._run_path / "models"

    def get_reports_dir(self) -> Path:
        """retorna o diretório de relatórios."""
        return self._run_path / "reports"

    def get_explainability_dir(self) -> Path:
        """retorna o diretório de explicabilidade."""
        return self._run_path / "explainability"

    def get_logs_dir(self) -> Path:
        """retorna o diretório de logs."""
        return self._run_path / "logs"

    def metadata_path(self) -> Path:
        """retorna o caminho do arquivo de metadados."""
        return self._run_path / "metadata.json"

    def _create_metadata(self) -> None:
        """cria o arquivo metadata.json da execução atual."""

        metadata = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "status": "RUNNING",
            "pipeline_version": "2.0.0",
        }

        with self.metadata_path().open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

    def update_status(
        self,
        status: str,
    ) -> None:
        """
        atualiza o status da execução no metadata.json.

        parameters
        ----------
        status : str
            status da execução, como running, success ou failed.
        """

        metadata = {}

        if self.metadata_path().exists():
            with self.metadata_path().open(
                mode="r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        metadata["status"] = status
        metadata["finished_at"] = datetime.now().isoformat()

        with self.metadata_path().open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )