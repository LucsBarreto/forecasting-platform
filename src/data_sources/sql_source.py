"""
fonte de dados sql.

responsável pela leitura de dados provenientes de fontes sql.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.core.exceptions import DataLoadingError


class SqlSource:
    """lê dados de fontes sql."""

    def read(
        self,
        query: str,
        connection: Any,
    ) -> pd.DataFrame:
        """
        lê dados de uma fonte sql.

        parameters
        ----------
        query
            consulta sql a ser executada.

        connection
            conexão com o banco ou objeto compatível com sqlalchemy.

        returns
        -------
        pd.dataframe
            dados retornados pela consulta sql.

        raises
        ------
        dataloadingerror
            se a consulta sql não puder ser executada.
        """

        try:
            return pd.read_sql(
                query,
                connection,
            )

        except Exception as exc:
            raise DataLoadingError(
                "não foi possível ler os dados do banco"
            ) from exc