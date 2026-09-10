"""
excessões relacionadas com dados
"""

from .base import ApplicationError


class DataError(ApplicationError):
    """
    excessão base de dados
    """


class DataLoadingError(DataError):
    default_message = "dataset não foi carregado"


class UnsupportedFileExtensionError(DataError):
    default_message = "extensão de arquivo não suportada"


class EmptyDatasetError(DataError):
    default_message = "dataset esta vazio"


class DirectoryNotFoundError(DataError):
    default_message = "pasta não encontrada"