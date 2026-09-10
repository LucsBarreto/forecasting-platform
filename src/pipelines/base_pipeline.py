"""
abstração base dos pipelines.

este módulo fornece a interface comum e o comportamento de logging
dos pipelines da aplicação.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from src.core.logger.logger import get_logger


class BasePipeline(ABC):
    """
    classe base para os pipelines da aplicação.

    responsabilidades
    ------------------
    - definir a interface dos pipelines.
    - fornecer um logger para o pipeline.
    - registrar o tempo de execução do pipeline.
    - registrar falhas durante a execução.
    """

    def __init__(self) -> None:
        self.logger = get_logger()

    @property
    @abstractmethod
    def name(self) -> str:
        """retorna o nome do pipeline."""

    @abstractmethod
    def run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """executa o pipeline."""

    def _log_start(self) -> float:
        """
        registra o início da execução do pipeline.

        returns
        -------
        float
            timestamp de início da execução.
        """

        self.logger.info(
            f"começando o pipeline: {self.name}"
        )

        return perf_counter()

    def _log_finish(
        self,
        start_time: float,
    ) -> None:
        """registra a conclusão da execução do pipeline."""

        elapsed = perf_counter() - start_time

        self.logger.success(
            f"pipeline '{self.name}' finalizado "
            f"em {elapsed:.2f} segundos"
        )

    def _log_failure(
        self,
        exc: Exception,
    ) -> None:
        """registra uma falha durante a execução do pipeline."""

        self.logger.exception(exc)