"""
registry de leitores de dados.

mantém o mapeamento entre extensões de arquivos
e seus respectivos leitores de dados.
"""

from __future__ import annotations

from typing import Type

from .base import BaseSource


class DataSourceRegistry:
    """gerencia os leitores de dados registrados."""

    def __init__(self) -> None:
        """inicializa um registro vazio."""

        self._readers: dict[str, Type[BaseSource]] = {}

    def register(
        self,
        extension: str,
        reader: Type[BaseSource],
    ) -> None:
        """
        registra um leitor para uma extensão de arquivo.

        parameters
        ----------
        extension
            extensão do arquivo, como ".csv".

        reader
            classe responsável pela leitura dos dados.
        """

        normalized_extension = extension.lower()

        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        self._readers[normalized_extension] = reader

    def get(
        self,
        extension: str,
    ) -> Type[BaseSource]:
        """
        retorna o leitor registrado para uma extensão.

        parameters
        ----------
        extension
            extensão do arquivo.

        returns
        -------
        type[basesource]
            classe do leitor registrado.

        raises
        ------
        keyerror
            se nenhum leitor estiver registrado para a extensão.
        """

        normalized_extension = extension.lower()

        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        return self._readers[normalized_extension]

    def contains(
        self,
        extension: str,
    ) -> bool:
        """verifica se uma extensão está registrada."""

        normalized_extension = extension.lower()

        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        return normalized_extension in self._readers

    def unregister(
        self,
        extension: str,
    ) -> None:
        """
        remove um leitor do registro.

        parameters
        ----------
        extension
            extensão do arquivo.
        """

        normalized_extension = extension.lower()

        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        self._readers.pop(normalized_extension, None)

    def clear(self) -> None:
        """remove todos os leitores registrados."""

        self._readers.clear()