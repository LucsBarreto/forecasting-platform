"""
excessões de configuração
"""

from .base import ApplicationError


class ConfigurationError(ApplicationError):
    """
    excessão de configuração básica
    """


class MissingConfigurationError(ConfigurationError):
    default_message = "arquivo de configuração não encontrado"


class InvalidConfigurationError(ConfigurationError):
    default_message = "configuração invalida"