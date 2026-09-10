"""
utilitários para exportação de previsões.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.core.exceptions.data import DataError


@dataclass(slots=True)
class ForecastExporter:
    """
    exporta resultados de previsão.

    responsabilidades
    ------------------
    - validar os dados de previsão.
    - criar diretórios de saída.
    - exportar os resultados de previsão.
    """

    output_directory: Path

    def export_csv(
        self,
        dataframe: pd.DataFrame,
        filename: str = "forecast.csv",
    ) -> Path:
        """exporta os resultados de previsão para csv."""

        self._validate_dataframe(dataframe)
        filename = self._validate_filename(filename)

        output_path = self._prepare_output_path(
            filename,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        return output_path

    def export_parquet(
        self,
        dataframe: pd.DataFrame,
        filename: str = "forecast.parquet",
    ) -> Path:
        """exporta os resultados de previsão para parquet."""

        self._validate_dataframe(dataframe)
        filename = self._validate_filename(filename)

        output_path = self._prepare_output_path(
            filename,
        )

        dataframe.to_parquet(
            output_path,
            index=False,
        )

        return output_path

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:
        """valida o dataframe de previsão."""

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:
            raise DataError(
                "Cannot export an empty forecast dataframe."
            )

    @staticmethod
    def _validate_filename(
        filename: str,
    ) -> str:
        """valida o nome do arquivo de saída."""

        if not isinstance(
            filename,
            str,
        ) or not filename.strip():
            raise DataError(
                "Output filename must be a non-empty string."
            )

        path = Path(filename)

        if path.name != filename:
            raise DataError(
                "Output filename must not contain directories."
            )

        return filename

    def _prepare_output_path(
        self,
        filename: str,
    ) -> Path:
        """cria o diretório de saída e retorna o caminho do arquivo."""

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self.output_directory / filename