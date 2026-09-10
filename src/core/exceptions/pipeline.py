"""
excessões de pipeline
"""

from .base import ApplicationError


class PipelineError(ApplicationError):
    """
    excessões de pipeline basicas
    """


class PipelineExecutionError(PipelineError):
    default_message = "execução do pipeline falhou"