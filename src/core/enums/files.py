"""
tipos de arquivo suportados pela plataforma.
"""

from enum import StrEnum


class FileType(StrEnum):
    """extensões de arquivos suportadas."""

    XLSX = ".xlsx"

    XLSB = ".xlsb"

    CSV = ".csv"

    PARQUET = ".parquet"