"""
leitor de excel.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.config import settings
from src.core.exceptions import DataLoadingError
from src.core.logger.logger import get_logger

from .base import BaseSource

logger = get_logger()


def resolve_excel_engine(source: Path | BinaryIO, suffix: str) -> str | None:
    """seleciona um leitor de excel pelo conteúdo do arquivo e não pela extensão."""

    normalized_suffix = suffix.lower()
    if normalized_suffix != ".xlsb":
        return "openpyxl" if normalized_suffix in {".xlsx", ".xlsm"} else None

    try:
        if isinstance(source, Path):
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
        else:
            position = source.tell()
            source.seek(0)
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
            source.seek(position)
    except (OSError, zipfile.BadZipFile):
        return "pyxlsb"

    if "xl/workbook.bin" in names:
        return "pyxlsb"
    if "xl/workbook.xml" in names:
        return "openpyxl"
    return "pyxlsb"


class ExcelSource(BaseSource):
    """leitor de excel"""

    def read(self, source: Path) -> pd.DataFrame:
        """
        lê um arquivo em excel.

        args:
            caminho pro arquivo excel.

        returns:
            dataframe
        """

        logger.info(f"Lendo arquivo excel: {source}")

        try:
            df = pd.read_excel(
                source,
                sheet_name = settings.data.sheet_name,
                engine=resolve_excel_engine(
                    source,
                    source.suffix,
                ),
            )

            logger.success(
                f"foram carregadas {len(df):,} linhas de {source.name}"
            )

            return df

        except Exception as exc:
            logger.exception(exc)

            raise DataLoadingError(
                f"o arquivo não pode ser lido: {source}"
            ) from exc
