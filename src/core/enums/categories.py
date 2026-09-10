"""
categorias das regras de validação.
"""

from enum import StrEnum


class ValidationCategory(StrEnum):
    """categorias suportadas pelas regras de validação."""

    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    BUSINESS = "business"
    STATISTICAL = "statistical"