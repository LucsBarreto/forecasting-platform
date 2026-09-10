"""
pipeline de engenharia de features.

este módulo orquestra todas as etapas de engenharia de features.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.feature_engineering.business import BusinessFeatureEngineer
from src.feature_engineering.lag import LagFeatureEngineer
from src.feature_engineering.rolling import RollingFeatureEngineer
from src.feature_engineering.temporal import TemporalFeatureEngineer
from src.feature_engineering.trend import TrendFeatureEngineer


@dataclass(slots=True)
class FeatureEngineeringPipeline:
    """
    executa as etapas de engenharia de features na ordem configurada.

    o pipeline é responsável apenas pela orquestração.
    a lógica de criação das features permanece em cada engineer.
    """

    temporal: TemporalFeatureEngineer
    lag: LagFeatureEngineer
    rolling: RollingFeatureEngineer
    trend: TrendFeatureEngineer
    business: BusinessFeatureEngineer

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa todas as etapas de engenharia de features.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe enriquecido com todas as features configuradas.
        """

        data = dataframe.copy()

        data = self.temporal.process(data)

        data = self.lag.process(data)

        data = self.rolling.process(data)

        data = self.trend.process(data)

        data = self.business.process(data)

        return data