"""
excessões base pro aplicativo
"""


class ApplicationError(Exception):
    """
    excessão base pro projeto
    """

    default_message = "Application error."

    def __init__(
        self,
        message: str | None = None,
    ) -> None:

        super().__init__(
            message or self.default_message
        )