"""
processador de conversão de tipos de dados.

este módulo é responsável por converter as colunas do dataframe
para os tipos de dados configurados.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import settings
from src.core.exceptions import DataValidationError
from src.core.utils.datetime import coerce_excel_datetime


@dataclass(slots=True)
class TypeConverter:
    """
    converte as colunas do dataframe para os tipos de dados configurados.

    responsabilidades
    ------------------
    - identificar os tipos de dados configurados para cada coluna.
    - converter as colunas para os tipos de dados especificados.
    - tratar conversões de datas e horas.
    - reportar erros de validação durante as conversões.
    """

    def process(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        executa o pipeline de conversão de tipos.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe com os tipos de dados convertidos.
        """

        data = dataframe.copy()

        dtypes = settings.data.schema_config.dtypes

        for column, dtype in dtypes.items():

            if column not in data.columns:
                continue

            data[column] = self._convert_column(
                data[column],
                dtype,
                column,
            )

        return data

    @staticmethod
    def _convert_column(
        series: pd.Series,
        dtype: str,
        column_name: str,
    ) -> pd.Series:
        """
        converte uma única coluna do dataframe para o tipo configurado.

        operations
        ----------
        - converte valores para texto.
        - converte valores para categorias.
        - converte valores para inteiros.
        - converte valores para números de ponto flutuante.
        - converte valores para booleanos.
        - converte valores para data e hora.
        - aplica outros tipos de dados configurados.
        - gera erro de validação quando a conversão falha.
        """

        try:

            match dtype:

                case "string":
                    return series.astype("string")

                case "category":
                    return series.astype("category")

                case "int64":
                    return pd.to_numeric(
                        series,
                        errors="raise",
                    ).astype("int64")

                case "float64":
                    return pd.to_numeric(
                        series,
                        errors="raise",
                    ).astype("float64")

                case "boolean":
                    return series.astype("boolean")

                case "datetime64[ns]":
                    result = coerce_excel_datetime(series)
                    if result.isna().any():
                        raise ValueError("invalid datetime values")
                    return result

                case _:
                    return series.astype(dtype)

        except Exception as exc:

            raise DataValidationError(

                    f"Unable to convert column "
                    f"'{column_name}' to '{dtype}'."

            ) from exc
