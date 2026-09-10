"""
métricas de profiling.

este módulo define as estruturas de dados utilizadas para representar
as métricas de execução.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """
    representa as métricas de profiling da execução.

    parameters
    ----------
    elapsed_seconds
        tempo de execução em segundos.

    memory_before_mb
        consumo de memória antes da execução, em megabytes.

    memory_after_mb
        consumo de memória após a execução, em megabytes.

    memory_delta_mb
        diferença entre o consumo de memória após e antes da execução.
    """

    elapsed_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float

    def to_dict(self) -> dict[str, float]:
        """
        converte as métricas para um dicionário.

        returns
        -------
        dict[str, float]
            dicionário contendo as métricas de execução.
        """

        return {
            "elapsed_seconds": self.elapsed_seconds,
            "memory_before_mb": self.memory_before_mb,
            "memory_after_mb": self.memory_after_mb,
            "memory_delta_mb": self.memory_delta_mb,
        }
