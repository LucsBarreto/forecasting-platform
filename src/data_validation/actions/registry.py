"""
registrador de ações de validação.
"""

from __future__ import annotations

from .base import BaseValidationAction
from src.core.enums import ValidationActionType


_ACTIONS: dict[
    ValidationActionType,
    type[BaseValidationAction],
] = {}


def register_action(
    action: ValidationActionType,
):
    """
    registra uma ação de validação.
    """

    def decorator(
        cls: type[BaseValidationAction],
    ) -> type[BaseValidationAction]:

        _ACTIONS[action] = cls

        return cls

    return decorator


def get_action(
    name: ValidationActionType | str,
) -> type[BaseValidationAction] | None:
    """
    retorna a ação registrada.
    """

    if isinstance(name, str):
        try:
            name = ValidationActionType(name.lower())
        except ValueError:
            return None

    return _ACTIONS.get(name)