"""
fábrica de ações de validação.
"""

from __future__ import annotations

from src.core.enums import ValidationActionType
from src.core.exceptions import InvalidConfigurationError

from .base import BaseValidationAction
from .registry import get_action


class ValidationActionFactory:
    """
    fábrica responsável por criar ações de validação.
    """

    @staticmethod
    def create(
        action: ValidationActionType,
    ) -> BaseValidationAction:

        action_class = get_action(action)

        if action_class is None:
            raise InvalidConfigurationError(
                f"Ação de validação não suportada: {action}"
            )

        return action_class()