"""
pipeline de pré-processamento.

este módulo orquestra os processadores responsáveis pelo
pré-processamento dos dados.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .cleaning import CleaningProcessor
from .missing import MissingValueProcessor
from .typing import TypeConverter
from .datetime import DatetimeProcessor


@dataclass(slots=True)
class PreprocessingPipeline:
    """
    orquestra as etapas de pré-processamento.

    responsabilidades
    ------------------
    - executar o processador de limpeza.
    - executar o processador de valores ausentes.
    - executar o processador de conversão de tipos.
    - executar o processador de criação de features temporais.

    ordem de execução
    -----------------
    1. limpeza.
    2. tratamento de valores ausentes.
    3. conversão de tipos.
    4. criação de features temporais.
    """

    cleaning: CleaningProcessor
    missing: MissingValueProcessor
    typing: TypeConverter
    datetime: DatetimeProcessor

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa o pipeline completo de pré-processamento.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe pré-processado.
        """

        data = dataframe.copy()

        data = self.cleaning.process(data)

        data = self.missing.process(data)

        data = self.typing.process(data)

        data = self.datetime.process(data)

        return data
