"""
factory de regras de validação.
"""

from __future__ import annotations

from src.core.enums import ValidationCategory

from src.data_validation.base import BaseValidationRule

from .registry import get_rules


class ValidationRuleFactory:
    """cria instâncias das regras de validação registradas."""

    @staticmethod
    def create(
        category: ValidationCategory | None = None,
    ) -> list[BaseValidationRule]:
        """cria instâncias das regras de validação."""

        rules = get_rules()

        if category is not None:

            rules = [
                metadata
                for metadata in rules
                if metadata.category == category
            ]

        return [
            metadata.rule()
            for metadata in rules
        ]