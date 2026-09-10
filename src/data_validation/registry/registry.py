"""
registro de regras de validação.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.enums import ValidationCategory

from src.data_validation.base import BaseValidationRule

from .metadata import RuleMetadata


_RULES: list[RuleMetadata] = []


def register_rule(
    *,
    order: int,
    category: ValidationCategory,
    enabled_by_default: bool = True,
):
    """registra uma regra de validação."""

    def decorator(
        cls: type[BaseValidationRule],
    ) -> type[BaseValidationRule]:

        _RULES.append(
            RuleMetadata(
                order=order,
                category=category,
                enabled_by_default=enabled_by_default,
                rule=cls,
            )
        )

        return cls

    return decorator


def get_rules() -> list[RuleMetadata]:
    """retorna as regras registradas ordenadas pela prioridade de execução."""

    return sorted(
        _RULES,
        key=lambda metadata: metadata.order,
    )