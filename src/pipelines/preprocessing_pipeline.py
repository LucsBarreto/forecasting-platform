"""
pipeline de pré-processamento.

este módulo orquestra todas as etapas de pré-processamento.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.aggregation.aggregator import Aggregator
from src.preprocessing.cleaning import CleaningProcessor
from src.preprocessing.missing import MissingValueProcessor
from src.preprocessing.typing import TypeConverter


@dataclass(slots=True)
class PreprocessingPipeline:
    """executa as etapas de pré-processamento na ordem configurada."""

    cleaning: CleaningProcessor
    missing: MissingValueProcessor
    typing: TypeConverter
    aggregation: Aggregator

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """executa todas as etapas de pré-processamento."""

        data = dataframe.copy()

        data = self.cleaning.process(data)

        data = self.missing.process(data)

        data = self.typing.process(data)

        data = self.aggregation.aggregate(data)

        return data