"""
leitor de csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.exceptions import DataLoadingError
from src.core.logger.logger import get_logger

from .base import BaseSource

logger = get_logger()

class CsvSource(BaseSource):
    """leitor de arquivos csv"""

    def read(self, source: Path) -> pd.DataFrame:

        logger.info(f"lendo arquivo csv: {source}")

        try:
            df = pd.read_csv(source)

            logger.success(
                f"foram carregadas {len(df):,} linhas do {source.name}"
            )

            return df

        except Exception as exc:
            logger.exception(exc)

            raise DataLoadingError(
                f"não foi possivel abrir o arquivo csv"
            ) from exc