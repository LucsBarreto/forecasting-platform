"""
descobre os arquivos de entrada da aplicação.
"""

from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.core.exceptions import DataLoadingError
from src.core.logger.logger import get_logger

logger = get_logger()


class FileDiscovery:
    """descobre arquivos com extensões compatíveis."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def discover(self) -> list[Path]:
        """
        localiza os arquivos com extensões suportadas.

        returns:
            lista de caminhos dos arquivos encontrados.

        raises:
            dataloadingerror:
                se o diretório não existir ou nenhum arquivo suportado for encontrado.
        """

        logger.info(f"procurando arquivos em: '{self._directory}'.")

        if not self._directory.exists():
            raise DataLoadingError(
                f"diretório não encontrado: {self._directory}"
            )

        if not self._directory.is_dir():
            raise DataLoadingError(
                f"'{self._directory}' não é um diretório."
            )

        files: list[Path] = []

        for extension in settings.data.supported_extensions:
            files.extend(
                sorted(self._directory.glob(f"*{extension}"))
            )

        if not files:
            raise DataLoadingError(
                "nenhum arquivo com extensão suportada foi encontrado."
            )

        logger.success(
            f"{len(files)} arquivos descobertos."
        )

        return files