"""
validador de schema.

este módulo é responsável por validar a estrutura e os tipos de dados
de um dataframe de acordo com a definição do schema.
"""

from __future__ import annotations

import pandas as pd

from src.core.exceptions import (
    InvalidColumnTypeError,
    MissingColumnError,
)
from src.core.logger.logger import get_logger

from .models import DataSchema

logger = get_logger()


class SchemaValidator:
    """
    valida um dataframe de acordo com a definição do schema.

    responsabilidades
    ------------------
    - validar a presença das colunas obrigatórias.
    - validar os tipos de dados das colunas configuradas.
    - gerar erros para colunas obrigatórias ausentes.
    - gerar erros para tipos de dados incompatíveis.
    - registrar o resultado da validação no logger.
    """

    def __init__(
        self,
        schema: DataSchema,
    ) -> None:
        self._schema = schema

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        valida a estrutura e os tipos de dados do dataframe.

        parameters
        ----------
        dataframe
            dataframe de entrada.

        returns
        -------
        pd.dataframe
            dataframe validado.

        raises
        ------
        missingcolumnerror
            se uma coluna obrigatória estiver ausente no dataframe.

        invalidcolumntypeerror
            se uma coluna não possuir o tipo de dado especificado.
        """

        logger.info("validando o schema do dataset")

        self._validate_required_columns(dataframe)

        self._validate_dtypes(dataframe)

        logger.success("validação concluida")

        return dataframe

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        valida a presença das colunas obrigatórias.

        parameters
        ----------
        dataframe
            dataframe que será validado.

        raises
        ------
        missingcolumnerror
            se uma coluna obrigatória estiver ausente no dataframe.
        """

        for column in self._schema.required_columns:

            if column not in dataframe.columns:
                raise MissingColumnError(column)

    def _validate_dtypes(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        valida os tipos de dados das colunas configuradas.

        parameters
        ----------
        dataframe
            dataframe que será validado.

        raises
        ------
        invalidcolumntypeerror
            se uma coluna não possuir o tipo de dado esperado.
        """

        for column, expected_dtype in self._schema.dtypes.items():

            if column not in dataframe.columns:
                continue

            current_dtype = str(
                dataframe[column].dtype
            )

            if current_dtype != expected_dtype:

                raise InvalidColumnTypeError(
                    column=column,
                    expected=expected_dtype,
                    received=current_dtype,
                )
