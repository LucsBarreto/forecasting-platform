"""
enums de validação.
"""

from enum import StrEnum


class ValidationActionType(StrEnum):
    """ações de validação de dados suportadas."""

    REMOVE = "remove"

    KEEP = "keep"

    WARN = "warn"