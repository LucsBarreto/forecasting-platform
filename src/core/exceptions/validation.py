"""
excessões de validação
"""

from .base import ApplicationError


class DataValidationError(ApplicationError):
    """
    excessões basicas de validação
    """


class MissingColumnError(DataValidationError):
    default_message = "coluna necessária não encontrada"


class InvalidColumnTypeError(DataValidationError):
    """quando uma coluna tem um tipo invalido"""

    def __init__(
        self,
        column: str,
        expected: str,
        received: str,
    ) -> None:

        self.column = column
        self.expected = expected
        self.received = received

        super().__init__(
            f"Column '{column}' has invalid dtype. "
            f"Expected '{expected}', received '{received}'."
        )


class InvalidDateColumnError(DataValidationError):
    default_message = "coluna de data invalida"


class SchemaValidationError(DataValidationError):
    default_message = "validação do schema falhou"