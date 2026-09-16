"""carrega contratos YAML do projeto."""

from pathlib import Path
from typing import Any

import yaml

from src.core.constants import PROJECT_ROOT
from src.core.exceptions import ConfigurationError


class ConfigLoader:
    """carrega contratos YAML em memória."""

    def __init__(self, config_directory: Path | str) -> None:
        base_directory = Path(config_directory)
        if not base_directory.is_absolute():
            base_directory = (PROJECT_ROOT / base_directory).resolve()
        self._config_directory = base_directory

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