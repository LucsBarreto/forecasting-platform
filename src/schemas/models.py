"""
modelos de schema.

este módulo define os modelos utilizados para representar
a estrutura e as regras de validação das colunas do dataset.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    """
    define a estrutura e as regras de uma coluna do dataset.

    responsabilidades
    ------------------
    - definir o nome da coluna.
    - definir o tipo de dado esperado.
    - indicar se a coluna é obrigatória.
    - indicar se valores nulos são permitidos.
    """

    name: str = Field(..., description="Column name.")

    dtype: str = Field(..., description="Expected pandas dtype.")

    required: bool = Field(
        default=True,
        description="Whether the column is mandatory.",
    )

    nullable: bool = Field(
        default=True,
        description="Whether null values are allowed.",
    )


class DataSchema(BaseModel):
    """
    define o schema esperado para o dataset.

    responsabilidades
    ------------------
    - armazenar as definições das colunas.
    - disponibilizar as colunas obrigatórias.
    - disponibilizar os tipos de dados esperados.
    """

    columns: list[ColumnSchema]

    @property
    def required_columns(self) -> list[str]:
        """
        retorna os nomes das colunas obrigatórias.

        returns
        -------
        list[str]
            lista contendo os nomes das colunas obrigatórias.
        """

        return [
            column.name
            for column in self.columns
            if column.required
        ]

    @property
    def dtypes(self) -> dict[str, str]:
        """
        retorna os tipos de dados esperados para cada coluna.

        returns
        -------
        dict[str, str]
            dicionário relacionando o nome da coluna ao tipo esperado.
        """

        return {
            column.name: column.dtype
            for column in self.columns
        }
