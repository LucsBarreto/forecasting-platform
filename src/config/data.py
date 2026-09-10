"""
configurações relacionadas aos dados.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HistorySettings(BaseModel):
    """configuração do histórico de dados."""

    minimum_years: int = 2


class SchemaSettings(BaseModel):
    """configuração do schema de dados."""

    dtypes: dict[str, str]


class DataSettings(BaseModel):
    """configuração de ingestão e armazenamento de dados."""

    input_path: str

    output_path: str

    runs_folder: str

    file_pattern: str

    required_columns: list[str]

    date_column: str

    history: HistorySettings

    granularity: list[str]

    schema_config: SchemaSettings = Field(
        alias="schema",
    )

    sheet_name: str | int = 0

    supported_extensions: tuple[str, ...] = (
        ".xlsx",
        ".xls",
        ".xlsb",
        ".csv",
        ".parquet",
    )