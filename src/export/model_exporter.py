"""
utilitários para exportação de modelos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.core.exceptions.modeling import ModelingError


@dataclass(slots=True)
class ModelExporter:
    """
    exporta modelos de machine learning treinados.

    responsabilidades
    ------------------
    - validar modelos.
    - criar diretórios de saída.
    - persistir modelos treinados.
    - carregar modelos persistidos.
    """

    output_directory: Path

    def save(
        self,
        model: Any,
        filename: str,
    ) -> Path:
        """persiste um modelo treinado."""

        self._validate_model(model)

        filename = self._validate_filename(
            filename,
        )

        output_path = self._prepare_output_path(
            filename,
        )

        joblib.dump(
            model,
            output_path,
        )

        return output_path

    def load(
        self,
        filename: str,
    ) -> Any:
        """carrega um modelo persistido."""

        filename = self._validate_filename(
            filename,
        )

        model_path = (
            self.output_directory / filename
        )

        if not model_path.exists():
            raise ModelingError(
                f"Model file was not found: {model_path}"
            )

        return joblib.load(
            model_path,
        )

    @staticmethod
    def _validate_model(
        model: Any,
    ) -> None:
        """valida o modelo antes da persistência."""

        if model is None:
            raise ModelingError(
                "Cannot export a null model."
            )

    @staticmethod
    def _validate_filename(
        filename: str,
    ) -> str:
        """valida o nome do arquivo do modelo."""

        if not isinstance(
            filename,
            str,
        ) or not filename.strip():
            raise ModelingError(
                "Model filename must be a non-empty string."
            )

        path = Path(filename)

        if path.name != filename:
            raise ModelingError(
                "Model filename must not contain directories."
            )

        return filename

    def _prepare_output_path(
        self,
        filename: str,
    ) -> Path:
        """cria o diretório de saída e retorna o caminho do arquivo."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.output_directory / filename