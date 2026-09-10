"""
escrita de dados.

responsável por persistir dataframes nos formatos de saída suportados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.exceptions import DataLoadingError


class DataWriter:
    """escreve dataframes nos formatos de arquivo suportados."""

    _WRITERS = {
        ".csv": "_write_csv",
        ".parquet": "_write_parquet",
        ".xlsx": "_write_excel",
    }

    def write(
        self,
        dataframe: pd.DataFrame,
        destination: Path,
    ) -> None:
        """
        escreve um dataframe no destino informado.

        parameters
        ----------
        dataframe
            dataframe a ser persistido.

        destination
            caminho do arquivo de destino.

        raises
        ------
        dataloadingerror
            se o formato do destino não for suportado ou ocorrer
            um erro durante a escrita.
        """

        extension = destination.suffix.lower()

        if extension not in self._WRITERS:
            raise DataLoadingError(
                f"formato de arquivo não suportado para escrita: {extension}"
            )

        writer_name = self._WRITERS[extension]
        writer = getattr(self, writer_name)

        try:
            writer(
                dataframe,
                destination,
            )

        except Exception as exc:
            raise DataLoadingError(
                f"não foi possível escrever o arquivo: {destination}"
            ) from exc

    @staticmethod
    def _write_csv(
        dataframe: pd.DataFrame,
        destination: Path,
    ) -> None:
        """grava o dataframe em csv."""

        dataframe.to_csv(
            destination,
            index=False,
        )

    @staticmethod
    def _write_parquet(
        dataframe: pd.DataFrame,
        destination: Path,
    ) -> None:
        """grava o dataframe em parquet."""

        dataframe.to_parquet(
            destination,
            index=False,
        )

    @staticmethod
    def _write_excel(
        dataframe: pd.DataFrame,
        destination: Path,
    ) -> None:
        """grava o dataframe em excel."""

        dataframe.to_excel(
            destination,
            index=False,
        )