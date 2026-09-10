"""
frequências suportadas pela plataforma.
"""

from enum import StrEnum


class FrequencyType(StrEnum):
    """frequências temporais suportadas."""

    DAILY = "D"

    WEEKLY = "W"

    MONTHLY = "M"

    QUARTERLY = "Q"

    YEARLY = "Y"