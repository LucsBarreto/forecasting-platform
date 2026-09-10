"""
leitor de arquivo parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.exceptions import DataLoadingError
from src.core.logger.logger import get_logger

from .base import BaseSource

logger = get_logger()

class ParquetSource(BaseSource):
    """leitor de arquivos parquet"""

    def read(self, source: Path) -> pd.DataFrame:

        logger.info(f"lendo arquivo parquet: {source}")

        try:
            df = pd.read_parquet(source)

            logger.success(
                f"foram carregadas {len(df):,} linhas do {source.name}"
            )

            return df

        except Exception as exc:
            logger.exception(exc)

            raise DataLoadingError(
                f"não foi possivel abrir o arquivo parquet"
            ) from exc