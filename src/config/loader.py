"""carrega contratos YAML do projeto."""

from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ConfigurationError


class ConfigLoader:
    """carrega contratos YAML em memória."""

    def __init__(self, config_directory: Path) -> None:
        self._config_directory = config_directory

    def load(self, filename: str) -> dict[str, Any]:
        """lê um contrato YAML e retorna seus dados como dicionário."""

        file_path = self._config_directory / filename

        if not file_path.exists():
            raise ConfigurationError(f"arquivo não encontrado: {file_path}")

        try:
            with file_path.open(mode="r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"YAML inválido: {filename}") from exc

        return data or {}