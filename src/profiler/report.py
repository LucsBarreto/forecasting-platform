"""
relatórios de profiling.

este módulo transforma métricas de profiling em estruturas
adequadas para geração de relatórios.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.profiler.metrics import ProfileMetrics


@dataclass(slots=True)
class ProfileReport:
    """
    gera relatórios a partir das métricas de profiling.

    responsabilidades
    ------------------
    - disponibilizar as métricas em formato de dicionário.
    - disponibilizar as métricas em formato de dataframe.
    """

    metrics: ProfileMetrics

    def to_dict(self) -> dict[str, float]:
        """
        retorna as métricas em formato de dicionário.

        returns
        -------
        dict[str, float]
            dicionário contendo as métricas de profiling.
        """

        return self.metrics.to_dict()

    def to_dataframe(self) -> pd.DataFrame:
        """
        retorna as métricas em formato de dataframe.

        returns
        -------
        pd.dataframe
            dataframe contendo as métricas de profiling.
        """

        return pd.DataFrame(
            [
                self.to_dict(),
            ]
        )
