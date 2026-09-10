"""
classes básicas para os leitores de dados.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

class BaseSource(ABC):
    """classe abstrata base para os leitores de dados."""

    @abstractmethod
    def read(self, source: Path) -> pd.DataFrame:
        """
        lê os dados da fonte.

        args:
            caminho para o arquivo ou conexão.

        returns:
            dataframe com os dados lidos.
        """
        raise NotImplementedError