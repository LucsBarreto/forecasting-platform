"""
profiler de execução.

este módulo fornece utilitários para medir o tempo de execução
e o consumo de memória.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from src.profiler.metrics import ProfileMetrics


@dataclass(slots=True)
class Profiler:
    """
    mede o tempo de execução e o consumo de memória.

    responsabilidades
    ------------------
    - iniciar o monitoramento da execução.
    - medir o tempo de execução.
    - medir o consumo de memória antes e após a execução.
    - calcular a diferença de consumo de memória.
    - disponibilizar as métricas da execução.
    - permitir o uso como gerenciador de contexto.
    """

    _start_time: float | None = None
    _memory_before: int | None = None
    _metrics: ProfileMetrics | None = None

    def start(self) -> None:
        """
        inicia o profiling da execução.
        """

        if self._start_time is not None:
            raise RuntimeError(
                "Profiler is already running."
            )

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        current, _ = tracemalloc.get_traced_memory()

        self._start_time = time.perf_counter()
        self._memory_before = current
        self._metrics = None

    def stop(self) -> ProfileMetrics:
        """
        encerra o profiling e retorna as métricas coletadas.

        returns
        -------
        ProfileMetrics
            métricas de tempo e consumo de memória da execução.
        """

        if self._start_time is None:
            raise RuntimeError(
                "Profiler has not been started."
            )

        current, _ = tracemalloc.get_traced_memory()

        elapsed = (
            time.perf_counter()
            - self._start_time
        )

        memory_before = self._memory_before or 0

        memory_before_mb = (
            memory_before
            / 1024
            / 1024
        )

        memory_after_mb = (
            current
            / 1024
            / 1024
        )

        memory_delta_mb = (
            memory_after_mb
            - memory_before_mb
        )

        self._metrics = ProfileMetrics(
            elapsed_seconds=elapsed,
            memory_before_mb=memory_before_mb,
            memory_after_mb=memory_after_mb,
            memory_delta_mb=memory_delta_mb,
        )

        self._start_time = None
        self._memory_before = None

        return self._metrics

    @property
    def metrics(self) -> ProfileMetrics | None:
        """
        retorna as métricas mais recentes do profiling.

        returns
        -------
        ProfileMetrics | None
            métricas mais recentes ou none caso nenhuma execução
            tenha sido finalizada.
        """

        return self._metrics

    @contextmanager
    def measure(self) -> Iterator[Profiler]:
        """
        realiza o profiling de um bloco de código.

        examples
        --------
        with profiler.measure():
            execute_pipeline()
        """

        self.start()

        try:
            yield self
        finally:
            self.stop()
